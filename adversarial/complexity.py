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
    non_batch_dims = list(range(2, len(x_left.shape)))
    square_dists = tf.reduce_sum(delta_square_per_dim, axis=non_batch_dims)
    avg_dists = tf.reduce_mean(square_dists)
    return avg_dists

@tf.function
def ce_loss(label, y):
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(label, y))

@tf.function
def projection(x, x_0, epsilon, inf_dataset, sup_dataset):
    x = x_0 + tf.clip_by_norm(x - x_0, epsilon)  # return to epsilon ball
    # x = tf.clip_by_value(x, inf_dataset, sup_dataset)  # return to image manifold
    return x

@tf.function
def gradient_step(model, label, x_0, x,
                  step_size, epsilon, lbda,
                  inf_dataset, sup_dataset):
    y = model(x)
    criterion = ce_loss(label, y)
    variance = variance_loss(x)
    loss = criterion + lbda * variance
    tf.print(criterion, variance, loss)
    g = tf.gradients(loss, [x])[0]
    x = x + step_size * g  # add gradient (Gradient Ascent)
    x = projection(x, x_0, epsilon, inf_dataset, sup_dataset)
    # tf.print(x, x - x_0, x_0, sep='\n')
    return x

@tf.function
def generate_population(x, label, epsilon, population_size):
    x_0 = tf.broadcast_to(x, shape=[population_size]+list(x.shape[1:]))
    label = tf.broadcast_to(label, shape=[population_size])
    x = tf.random.normal(x_0.shape, x_0, epsilon)
    return x, x_0, label

# @tf.function
def projected_gradient(model, x, label,
                       num_steps, step_size, population_size,
                       lbda, epsilon, inf_dataset, sup_dataset):
    x, x_0, label = generate_population(x, label,
                                        epsilon, population_size)
    x = projection(x, x_0, epsilon, inf_dataset, sup_dataset)
    for _ in range(num_steps):
        x = gradient_step(model, label, x_0, x,
                          step_size, epsilon, lbda,
                          inf_dataset, sup_dataset)
    return ce_loss(label, model(x))


def adversarial_score(model, dataset, num_batchs_max,
                      num_steps, step_size, population_size,
                      lbda, epsilon, inf_dataset, sup_dataset):
    losses = []
    for (x, label), _ in zip(dataset, progress_bar(num_batchs_max)):
        print('New ascent !!')
        print('Sizes: ', tf.reduce_max(x), tf.reduce_min(x), tf.reduce_mean(x))
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
    num_steps       = tf.constant(30, dtype=tf.int32)
    step_size       = tf.constant(1., dtype=tf.float32)
    population_size = 16
    lbda            = tf.constant(5., dtype=tf.float32)
    length_unit     = sqrt(float(tf.size(dummy_input)))
    epsilon         = tf.constant(0.05 * length_unit, dtype=tf.float32)
    inf_dataset     = tf.constant(-2., dtype=tf.float32)
    sup_dataset     = tf.constant(2., dtype=tf.float32)
    avg_loss = adversarial_score(model, dataset, num_batchs_max,
                                 num_steps, step_size, population_size,
                                 lbda, epsilon, inf_dataset, sup_dataset)
    return avg_loss


