"""Manifold Mixup Score.

python3 ingestion_program/ingestion_tqdm.py sample_data sample_result_submission ingestion_program adversarial
python3 scoring_program/score.py sample_data sample_result_submission scores

python3 ingestion_program/ingestion_tqdm.py ../datasets/public_data sample_result_submission ingestion_program adversarial
python3 scoring_program/score.py ../datasets/public_data sample_result_submission scores"""

import os
import numpy as np
import tensorflow as tf
from utils import progress_bar, balanced_batchs


@tf.function
def variance(x):  # norm2 distance squared
    x_left = tf.expand_dims(x, axis=1)
    x_right = tf.expand_dims(x, axis=0)
    delta_square_per_dim = (x_left - x_right) ** 2.
    non_batch_dims = list(range(2, len(x_left.shape)))
    square_dists = tf.reduce_sum(delta_square_per_dim, axis=non_batch_dims)
    avg_dists = tf.reduce_mean(square_dists)
    return avg_dists

@tf.function
def criterion(label, y):
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(label, y))

@tf.function
def gradient_step(model, label, x_0, x,
                  step_size, lbda=1.,
                  epsilon=0.3, inf_dataset=0., sup_dataset=1.):
    tf.print(x.shape)
    y = model(x)
    ce_loss = criterion(label, y)
    variance_loss = variance(x)
    loss = ce_loss  # + lbda * variance_loss
    tf.print(ce_loss, variance_loss, loss)
    tf.print(loss)
    g = tf.gradients(loss, [x])[0]
    x = x + step_size * g  # add gradient (Gradient Ascent)
    x = x_0 + tf.clip_by_norm(x - x_0, epsilon)  # return to epsilon ball
    x = tf.clip_by_value(x, inf_dataset, sup_dataset)  # return to image manifold
    # tf.print(x, x - x_0, x_0, sep='\n')
    return x

# @tf.function
def generate_population(x, label, step_size, population_size=32):
    x = tf.broadcast_to(x, shape=[population_size]+list(x.shape[1:]))
    label = tf.broadcast_to(label, shape=[population_size])
    x = tf.random.normal(x.shape, x, step_size)
    return x, label

# @tf.function
def projected_gradient(model, x, label, num_steps=10, step_size=0.1):
    x, label = generate_population(x, label, step_size)
    x_0 = x
    for _ in range(num_steps):
        x = gradient_step(model, label, x_0, x, step_size)
    return criterion(label, model(x))


def adversarial_score(model, dataset, num_batchs_max):
    losses = []
    print('New ascent !!')
    for (x, label), _ in zip(dataset, progress_bar(num_batchs_max)):
        pg_loss = projected_gradient(model, x, label)
        print('Descent over', pg_loss, '\n')
    return float(tf.reduce_mean(losses))

def complexity(model, dataset):
    output_shape = model(next(dataset.take(1).batch(1).__iter__())[0]).shape
    num_labels = int(output_shape[-1])
    debug = True
    num_batchs_max = 12 if debug else 256
    dataset = balanced_batchs(dataset, num_labels, 1)
    avg_loss = adversarial_score(model, dataset, num_batchs_max)
    return avg_loss


