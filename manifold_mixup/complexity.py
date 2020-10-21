"""Manifold Mixup Score.

python3 ingestion_program/ingestion_tqdm.py sample_data sample_result_submission ingestion_program manifold_mixup
python3 scoring_program/score.py sample_data sample_result_submission scores

python3 ingestion_program/ingestion_tqdm.py ../datasets/public_data sample_result_submission ingestion_program manifold_mixup
python3 scoring_program/score.py ../datasets/public_data sample_result_submission scores"""

import os
import numpy as np
import tensorflow as tf
from utils import *



#######################################################################
############################### Lipschitz #############################
#######################################################################

@tf.function
def evaluate_lip(model, x, labels, softmax):
    with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(x)
        y = model(x)
        if softmax:
            y = tf.nn.softmax(y)
            y = tf.gather(y, indices=labels, axis=1)  # axis=0 is batch dimension, axis=1 is logits
        else:
            y = tf.nn.softmax_cross_entropy_with_logits(label, y)
    dy_dx = tape.batch_jacobian(y, x)
    batch_squared_norm = tf.math.reduce_sum(dy_dx ** 2, axis=list(range(1,len(dy_dx.shape))))
    grad_penalty = batch_squared_norm
    lips = tf.math.reduce_mean(grad_penalty)
    return lips

def lipschitz_score(model, dataset, batch_size, num_batchs_max, softmax):
    dataset = raw_batchs(dataset, batch_size)  # balanced_batchs(dataset, num_labels, batch_size)
    scores = []
    progress = progress_bar(num_batchs_max)
    for (x, labels), _ in zip(dataset, progress):
        score = evaluate_lip(model, x, labels, softmax)
        scores.append(score)
    return float(tf.math.reduce_mean(scores))

@tf.function
def generate_gaussian_batch(noisy_per_epsilon, epsilon, x):
    no_batch_shape = tuple(x.shape[1:])
    noise_shape = (epsilon.shape[0],noisy_per_epsilon) + no_batch_shape
    mean = tf.reshape(x, shape=(1,1)+no_batch_shape)
    std = tf.reshape(epsilon, shape=(-1,1)+(1,)*len(no_batch_shape))
    batch = tf.random.normal(noise_shape, mean, std, dtype=tf.float32)
    output_shape = (-1,) + no_batch_shape
    return tf.reshape(batch, shape=output_shape)

@tf.function
def local_gaussian_is_robustness(x, batch, label, y, epsilon):
    no_batch_dims = list(range(-len(x.shape)+1,0))
    z = tf.math.reduce_sum((batch - x)**2, axis=no_batch_dims)
    z = tf.reshape(z, shape=(epsilon.shape[0],-1)) # shape E x K
    z = z / tf.reshape(epsilon, shape=(-1,1)) # warning, shape E x K
    z_source = tf.expand_dims(z, axis=1)  # sampled from
    z_target = tf.expand_dims(z, axis=0)  # used for
    is_weight = z_source - z_target  # shape E x E x K
    is_weight = tf.math.exp(0.5 * is_weight)  # shape E x E x K
    eps_source = tf.reshape(epsilon, axis=(1,-1))
    eps_target = tf.reshape(epsilon, axis=(-1,1))
    prefactor = tf.math.sqrt(eps_source / eps_target)  # warning
    is_weight = is_weight * prefactor
    error = tf.math.not_equal(label, tf.math.argmax(y, axis=-1))
    error = tf.dtypes.cast(error, dtype=tf.float32)  # shape E x K
    error = tf.expand_dims(error, axis=1)  # shape 1 x E x K because evaluated in source
    expectation = error * is_weight  # shape E x E x K
    expectation = tf.math.reduce_mean(expectation, axis=[-1,-2])
    return expectation  # for each epsilon, a result

@tf.function
def local_gaussian_robustness(batch, label, y, epsilon):
    # error = tf.math.not_equal(tf.math.argmax(y, axis=-1, output_type=tf.int32), label)
    label = tf.broadcast_to(label, y.shape)
    error = tf.nn.softmax_cross_entropy_with_logits(label, y)
    error = tf.dtypes.cast(error, tf.float32)  # shape (EK)
    error = tf.reshape(error, shape=(epsilon.shape[0],-1))  # shape (E, K)
    expectation = tf.math.reduce_mean(error, axis=-1)
    return expectation  # for each epsilon, a result

def rank_to_score(mean_per_eps):
    return float(tf.reduce_mean(mean_per_eps))

def mean_robustness(model, dataset, num_batchs_max, noisy_per_epsilon):
    dataset = raw_batchs(dataset, batch_size=1)  
    progress = progress_bar(num_batchs_max)
    epsilon = tf.constant([1e-2, 2e-2])
    robustnesses = []
    for (x, label), _ in zip(dataset, progress):
        batch = generate_gaussian_batch(noisy_per_epsilon, epsilon, x)
        y = model(batch)
        robustness = local_gaussian_robustness(batch, label, y, epsilon)
        robustnesses.append(robustness)
    mean_per_eps = tf.reduce_mean(robustnesses, axis=0)  # reduce on columns
    score = rank_to_score(mean_per_eps)
    return score

#######################################################################
############################# Complexity ##############################
#######################################################################

def complexity(model, dataset):
    # model.summary()
    public_data = False
    num_batchs_max = 1024
    batch_size = 12
    avg_loss = lipschitz_score(model, dataset, batch_size, num_batchs_max, softmax=True)
    # avg_loss = mixup_score(model, dataset, num_batchs_max, mix_policy='input')
    # avg_loss = catastrophic(model, dataset, num_batchs_witness=num_batchs_max, num_dumb_batchs=4)
    # avg_loss = graph_lip(model, dataset, num_batchs_max, almost_k_regular=8, layer_cut=-1, input_mixup=False)
    # avg_loss = mean_robustness(model, dataset, num_batchs_max, noisy_per_epsilon=16)
    return avg_loss
