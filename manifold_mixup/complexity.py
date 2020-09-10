"""Manifold Mixup Score.

python3 ingestion_program/ingestion_tqdm.py sample_data sample_result_submission ingestion_program manifold_mixup

python3 ingestion_program/ingestion_tqdm.py ../datasets/public_data sample_result_submission ingestion_program manifold_mixup

python3 scoring_program/score.py sample_data sample_result_submission scores

python3 scoring_program/score.py ../datasets/public_data sample_result_submission scores"""

import os
import numpy as np
import tensorflow as tf
try:
    raise 42
    import tqdm
    tqdm_pb = True
except:
    tqdm_pb = False


def raw_batchs(dataset, batch_size=256):
    return dataset.repeat().shuffle(buffer_size=10000).batch(batch_size)

def mixup_pairs(dataset):
    dataset = raw_batchs(dataset)
    for x, y in dataset:
        indexes = tf.random.shuffle(range(x.shape[0]))
        yield x, indexes, y, tf.gather(y, indexes)

@tf.function
def mix_manifold(model, x, indexes, lbda):
    def forward(layer, *inputs):
        if len(inputs) == 1:
            return layer(inputs[0])
        return layer(inputs[0]), layer(inputs[1])
    inputs = x, z
    for layer in model.layers:
        inputs = forward(layer, *inputs)
        if False:
            inputs = lbda*inputs[0] + (1.-lbda)*inputs[1]
    return inputs

@tf.function
def mix_input(model, x, indexes, lbda):
    z = tf.gather(x, indexes)
    mixed = lbda*x + (1.-lbda)*z
    mixed = model(mixed)
    return mixed

def criterion(logits, y):
    return tf.nn.sparse_softmax_cross_entropy_with_logits(y, logits)

def mixup_score(model, dataset, num_batchs_max, mix_policy, alpha=2.):
    losses = []
    mix_fn = mix_input if mix_policy == 'input' else mix_manifold
    progress = tqdm.tqdm(range(num_batchs_max), leave=False, ascii=True) if tqdm_pb else range(num_batchs_max)
    for (x, indexes, y, yt), _ in zip(mixup_pairs(dataset), progress):
        shape = (y.shape[0],) + (1,)*(len(x.shape)-1)
        lbda = tf.constant(np.random.beta(alpha, alpha, size=shape), dtype=tf.float32)
        mixed = mix_fn(model, x, indexes, lbda)
        loss = lbda*criterion(mixed, y) + (1.-lbda)*criterion(mixed, yt)
        loss = tf.math.reduce_mean(loss)
        losses.append(loss)
    return float(tf.math.reduce_mean(losses))

#######################################################################
############################### Lipschitz #############################
#######################################################################

def penalty(batch_squared_norm, one_lipschitz=False):
    batch_norm = tf.math.sqrt(batch_squared_norm)
    if one_lipschitz:
        return (batch_norm - 1.) ** 2
    return batch_norm

@tf.function
def evaluate_lip(model, x, labels):
    with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(x)
        y = model(x)
        y = tf.gather(y, indices=labels, axis=1)  # axis=0 is batch dimension, axis=1 is logits
    dy_dx = tape.gradient(y, x)
    batch_squared_norm = tf.math.reduce_sum(dy_dx ** 2, axis=list(range(1,len(dy_dx.shape))))
    grad_penalty = penalty(batch_squared_norm, one_lipschitz=True)
    lips = tf.math.reduce_mean(grad_penalty)
    return lips

def lipschitz_score(model, dataset, num_batchs_max):
    dataset = raw_batchs(dataset)
    scores = []
    progress = tqdm.tqdm(range(num_batchs_max), leave=False, ascii=True) if tqdm_pb else range(num_batchs_max)
    for (x, labels), _ in zip(dataset, progress):
        score = evaluate_lip(model, x, labels)
        scores.append(score)
    return float(tf.math.reduce_mean(scores))

def lipschitz_interpolation(model, dataset, num_batchs_max, alpha=2.):
    scores = []
    progress = tqdm.tqdm(range(num_batchs_max), leave=False, ascii=True) if tqdm_pb else range(num_batchs_max)
    for (x, indexes, labels, _), _ in zip(mixup_pairs(dataset), progress):
        shape = (labels.shape[0],) + (1,)*(len(x.shape)-1)
        lbda = tf.constant(np.random.beta(alpha, alpha, size=shape), dtype=tf.float32)
        z = tf.gather(x, indexes)
        mixed = lbda*x + (1.-lbda)*z
        score = evaluate_lip(model, mixed, labels)
        scores.append(score)
    return float(tf.math.reduce_mean(scores))

def complexity(model, dataset):
    # model.summary()
    num_batchs_max = 384
    avg_loss = lipschitz_interpolation(model, dataset, num_batchs_max, alpha=2.)
    # avg_loss = lipschitz_score(model, dataset, num_batchs_max)
    # avg_loss = mixup_score(model, dataset, num_batchs_max, mix_policy='input')
    return avg_loss