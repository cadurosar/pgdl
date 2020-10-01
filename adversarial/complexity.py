"""Manifold Mixup Score.

python3 ingestion_program/ingestion_tqdm.py sample_data sample_result_submission ingestion_program adversarial
python3 scoring_program/score.py sample_data sample_result_submission scores

python3 ingestion_program/ingestion_tqdm.py ../datasets/public_data sample_result_submission ingestion_program adversarial
python3 scoring_program/score.py ../datasets/public_data sample_result_submission scores"""

import os
from math import sqrt
import numpy as np
import tensorflow as tf
from utils import progress_bar, balanced_batchs


@tf.function
def variance_loss(x):  # norm2 distance squared
    x_left = tf.expand_dims(x, axis=1)
    x_right = tf.expand_dims(x, axis=0)
    delta_square_per_dim = (x_left - x_right) ** 2.
    non_batch_dims = list(range(2, len(delta_square_per_dim.shape)))
    square_dists = tf.reduce_sum(delta_square_per_dim, axis=non_batch_dims)
    avg_dists = tf.reduce_mean(square_dists)
    return avg_dists

@tf.function
def cosine_loss(x):  # norm2 distance squared
    non_batch_dims_norm = list(range(1, len(x.shape)))
    x_norm = tf.reduce_sum(x ** 2, axis=non_batch_dims_norm)
    x_norm_left = tf.expand_dims(x_norm, axis=1)
    x_norm_right = tf.expand_dims(x_norm, axis=0)
    x_left = tf.expand_dims(x, axis=1)
    x_right = tf.expand_dims(x, axis=0)
    dot_per_dim = x_left * x_right
    non_batch_dims = list(range(2, len(dot_per_dim.shape)))
    unnormalized = tf.reduce_sum(dot_per_dim, axis=non_batch_dims)
    cosine_sim = unnormalized / tf.sqrt(x_norm_left * x_norm_right)
    cosine_sim = tf.reduce_mean(cosine_sim)
    return -cosine_sim  # to be minimized

@tf.function
def ce_loss(label, y):
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(label, y))

@tf.function
def projection(x, x_0, epsilon, inf_dataset, sup_dataset):
    x = tf.clip_by_norm(x, epsilon)  # return to epsilon ball
    x = tf.clip_by_value(x + x_0, inf_dataset, sup_dataset) - x_0 # return to image manifold
    return x

@tf.function
def gradient_step(model, label, x_0, x,
                  step_size, epsilon, lbda,
                  inf_dataset, sup_dataset):
    y = model(x + x_0)
    criterion = ce_loss(label, y)
    # variance = variance_loss(x)
    variance = cosine_loss(x)
    loss = criterion + lbda * variance
    tf.print(criterion, variance, loss)
    g = tf.gradients(loss, [x])[0]
    x = x + step_size * g  # add gradient (Gradient Ascent)
    x = projection(x, x_0, epsilon, inf_dataset, sup_dataset)
    # tf.print(x, x - x_0, x_0, sep='\n')
    return x

#  @tf.function
def generate_population(x_0, label, epsilon, population_size):
    coordinate_wise = epsilon / sqrt(float(tf.size(x_0)))  # remove contribution to ball radius
    x_0 = tf.broadcast_to(x_0, shape=[population_size]+list(x_0.shape[1:]))
    label = tf.broadcast_to(label, shape=[population_size])
    x = tf.random.normal(x_0.shape, 0., coordinate_wise)  # within the ball
    # tf.print(x)
    return x, x_0, label

# @tf.function
def projected_gradient(model, x_0, label,
                       num_steps, step_size, population_size,
                       lbda, epsilon, inf_dataset, sup_dataset):
    x, x_0, label = generate_population(x_0, label,
                                        epsilon, population_size)
    x = projection(x, epsilon, inf_dataset, sup_dataset)
    for _ in range(num_steps):
        x = gradient_step(model, label, x_0, x,
                          step_size, epsilon, lbda,
                          inf_dataset, sup_dataset)
    return ce_loss(label, model(x + x_0))


def adversarial_score(model, dataset, num_batchs_max,
                      num_steps, step_size, population_size,
                      lbda, epsilon, inf_dataset, sup_dataset):
    losses = []
    for (x, label), _ in zip(dataset, progress_bar(num_batchs_max)):
        print('New ascent !!')
        # print('Sizes: ', tf.reduce_max(x), tf.reduce_min(x), tf.reduce_mean(x))
        pg_loss = projected_gradient(model, x, label,
                                     num_steps, step_size, population_size,
                                     lbda, epsilon, inf_dataset, sup_dataset)
        print('Ascent finished', pg_loss, '\n')
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
    num_batchs_max  = 256
    num_steps       = tf.constant(60, dtype=tf.int32)
    step_size       = tf.constant(1e-3, dtype=tf.float32)
    population_size = 8
    lbda            = tf.constant(1., dtype=tf.float32)
    length_unit     = sqrt(float(tf.size(dummy_input)))
    epsilon         = tf.constant(0.3 * length_unit, dtype=tf.float32)
    inf_dataset     = tf.constant(-2., dtype=tf.float32)
    sup_dataset     = tf.constant(2., dtype=tf.float32)
    avg_loss = adversarial_score(model, dataset, num_batchs_max,
                                 num_steps, step_size, population_size,
                                 lbda, epsilon, inf_dataset, sup_dataset)
    return avg_loss


