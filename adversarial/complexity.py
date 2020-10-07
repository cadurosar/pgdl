"""Adversarial score.

python3 ingestion_program/ingestion_tqdm.py sample_data sample_result_submission ingestion_program adversarial
python3 scoring_program/score.py sample_data sample_result_submission scores

python3 ingestion_program/ingestion_tqdm.py ../datasets/public_data sample_result_submission ingestion_program adversarial
python3 scoring_program/score.py ../datasets/public_data sample_result_submission scores"""

import os
import math
import numpy as np
import tensorflow as tf
from utils import progress_bar, balanced_batchs


@tf.function
def variance_loss(x):
    x_left                  = tf.expand_dims(x, axis=1)
    x_right                 = tf.expand_dims(x, axis=0)
    delta_square_per_dim    = (x_left - x_right) ** 2.
    non_batch_dims = list(range(2, len(delta_square_per_dim.shape)))
    square_dists            = tf.reduce_sum(delta_square_per_dim, axis=non_batch_dims)
    avg_dists               = tf.reduce_mean(square_dists)
    return avg_dists

@tf.function
def cosine_loss(x):
    x = tf.nn.avg_pool2d(x, ksize=[9, 9], strides=[5, 5],  # ensure that different regions are targeted
                         padding='SAME', data_format='NHWC')
    non_batch_dims_norm = list(range(1, len(x.shape)))
    x_norm          = tf.reduce_sum(x ** 2, axis=non_batch_dims_norm)
    x_norm_left     = tf.expand_dims(x_norm, axis=1)
    x_norm_right    = tf.expand_dims(x_norm, axis=0)
    x_left          = tf.expand_dims(x, axis=1)
    x_right         = tf.expand_dims(x, axis=0)
    dot_per_dim     = x_left * x_right
    non_batch_dims = list(range(2, len(dot_per_dim.shape)))
    unnormalized    = tf.reduce_sum(dot_per_dim, axis=non_batch_dims)
    unnormalized    = unnormalized ** 2.  # orthogonal is enough
    cosine_sim      = unnormalized / (x_norm_left * x_norm_right)
    cosine_sim      = tf.reduce_mean(cosine_sim)
    return -cosine_sim  # to be minimized

@tf.function
def ce_loss(label, y):
    threshold = tf.math.log(float(y.shape[-1]))
    full_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(label, y)
    full_loss = tf.minimum(full_loss, threshold)  # no need to further penalize
    full_loss = tf.reduce_mean(full_loss)
    return full_loss

@tf.function
def multi_targeted(label, y):
    depth = int(y.shape[-1])
    off_value = 1. / (depth-1.)
    multilabel = tf.one_hot(label, depth, on_value=0., off_value=off_value)
    multilabel_loss = tf.nn.softmax_cross_entropy_with_logits(multilabel, y)
    return -tf.reduce_mean(multilabel_loss)  # minimize equally adversary classes

@tf.function
def full_loss(label, y, x, lbda, euclidian_var):
    criterion = ce_loss(label, y)
    if euclidian_var:
        variance = variance_loss(x)
    else:
        variance = cosine_loss(x)
    loss = criterion + lbda * variance
    return loss, criterion, variance

@tf.function
def projection(x, x_0, epsilon, dataset_bounds):
    non_batch_dims = list(range(1, len(x.shape)))
    x = tf.clip_by_norm(x, epsilon, axes=non_batch_dims)  # return to epsilon ball
    x = tf.clip_by_value(x + x_0, dataset_bounds[0], dataset_bounds[1]) - x_0 # return to image manifold
    return x

@tf.function
def apply_gradient(x, g, x_0, step_size, epsilon, dataset_bounds):
    x = x + step_size * g  # Gradient Ascent
    x = projection(x, x_0, epsilon, dataset_bounds)
    return x

@tf.function
def gradient_step(model, label, x_0, x,
                  step_size, epsilon, lbda,
                  dataset_bounds,
                  euclidian_var):
    y                           = model(x + x_0)
    loss, criterion, variance   = full_loss(label, y, x, lbda, euclidian_var)
    g                           = tf.gradients(loss, [x])[0]
    x                           = apply_gradient(x, g, x_0, step_size, epsilon, dataset_bounds)
    return x, loss, criterion, variance

@tf.function
def generate_population(x_0, label, ball_l_inf, population_size):
    x_0             = tf.broadcast_to(x_0, shape=[population_size]+list(x_0.shape[1:]))
    label           = tf.broadcast_to(label, shape=[population_size])
    x               = tf.random.normal(x_0.shape, 0., ball_l_inf)  # within the ball
    return x, x_0, label

def projected_gradient(model, x_0, label,
                       num_steps, step_size, population_size,
                       lbda, epsilon, length_unit, sup_ce, dataset_bounds,
                       dilatation_rate, euclidian_var, verbose):
    ball_l_inf = epsilon / length_unit
    x, x_0, label = generate_population(x_0, label, ball_l_inf, population_size)
    x = projection(x, x_0, epsilon, dataset_bounds)
    tol_out = 0.5
    patience, tol_plateau = 5, 0.1
    last_plateau, last_criterion = 0, tf.constant(-math.inf)
    if verbose:
        print(' ',end='',flush=True)
        print(f'Start with radius {epsilon:.3f}')
    for step in range(num_steps):
        step_infos = gradient_step(model, label, x_0, x,
                                   step_size, epsilon, lbda,
                                   dataset_bounds,
                                   euclidian_var)
        x, loss, criterion, variance = step_infos
        if (verbose == 1 and step+1 == num_steps) or verbose == 2:
            print(f'Criterion={criterion:+5.3f} Variance={variance:+5.3f} Loss={loss:+5.3f}')
        if criterion - last_criterion >= tol_plateau * sup_ce:
            last_plateau = step
        last_criterion = criterion
        if step >= last_plateau+patience:
            x               = x * dilatation_rate
            epsilon         = epsilon * dilatation_rate
            x               = projection(x, x_0, epsilon, dataset_bounds)
            step_size       = step_size * dilatation_rate
            if verbose:
                print(f'Restart with radius {epsilon:.3f}')
            last_plateau    = step
            last_criterion  = tf.constant(-math.inf)
        if last_criterion >= tol_out * sup_ce:
            break  # optimal epsilon have been found
    return full_loss(label, model(x + x_0), x, lbda, euclidian_var), epsilon


def adversarial_score(model, dataset, num_batchs_max,
                      num_steps, step_size, population_size,
                      lbda, epsilon, length_unit, sup_ce, dataset_bounds,
                      dilatation_rate, euclidian_var, verbose):
    losses, radii = [], []
    for (x, label), _ in zip(dataset, progress_bar(num_batchs_max)):
        # print('Sizes: ', tf.reduce_max(x), tf.reduce_min(x), tf.reduce_mean(x))
        pg_results = projected_gradient(model, x, label,
                                        num_steps, step_size, population_size,
                                        lbda, epsilon, length_unit, sup_ce, dataset_bounds,
                                        dilatation_rate, euclidian_var, verbose)
        pg_loss, last_radius = pg_results
        losses.append(pg_loss)
        radii.append(last_radius)
    losses = tf.split(tf.stack(losses), num_or_size_splits=1)  # Median of Means
    losses = [tf.reduce_mean(loss) for loss in losses]
    losses = np.median([loss.numpy() for loss in losses])
    return float(tf.reduce_mean(losses))


def complexity(model, dataset):
    """Return complexity w.r.t a model and a dataset.

    Args:
        model: any Tensorflow model returning logits
        dataset: any TensorFlow Dataset with (x, label) pairs

    Return:
        the score

    Comments:
        num_batchs_max  = to be changed as function of running time
        num_steps       = path length
        step_size       = learning rate
        population_size = number of adversarial samples
        lbda            = weight of variance regularization
        length_unit     = length of vector (1,1,...,1)
        epsilon         = multiplicator of length_unit
        inf_dataset     = inf_dataset * (1. - sign(inf_dataset)*inf_dataset)
        sup_dataset     = sup_dataset * (1. + sign(sup_dataset)*sup_dataset)
    """
    dummy_input = next(dataset.take(1).batch(1).__iter__())[0]  # warning: one image disappears
    output_shape = model(dummy_input).shape
    num_labels = int(output_shape[-1])
    dataset         = balanced_batchs(dataset, num_labels, 1)  # one example at time
    num_batchs_max  = 320
    num_steps       = tf.constant(40, dtype=tf.int32)
    population_size = 4
    length_unit     = tf.math.sqrt(float(tf.size(dummy_input)))
    epsilon_mult    = 0.02
    epsilon         = tf.constant(epsilon_mult * length_unit, dtype=tf.float32)
    step_size       = tf.constant(5e-1, dtype=tf.float32)
    sup_ce          = tf.math.log(tf.constant(num_labels, dtype=tf.float32))
    lbda            = 0. * sup_ce  # normalize by typical magnitude
    euclidian_var   = False
    if euclidian_var:
        lbda        = lbda / (epsilon*epsilon)  # divide by average length
    inf_dataset     = tf.constant(-math.inf, dtype=tf.float32)
    sup_dataset     = tf.constant( math.inf, dtype=tf.float32)
    dilatation_rate = tf.constant(2.)
    verbose         = 2
    avg_loss = adversarial_score(model, dataset, num_batchs_max,
                                 num_steps, step_size, population_size,
                                 lbda, epsilon, length_unit, sup_ce,
                                 (inf_dataset, sup_dataset),
                                 dilatation_rate, euclidian_var, verbose)
    return avg_loss


