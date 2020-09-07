"""Manifold Mixup Score.

Command: python3 ingestion_program/ingestion.py ../datasets/public_data sample_result_submission ingestion_program manifold_mixup"""

import numpy as np
import tensorflow as tf


def raw_batchs(dataset, batch_size=128):
    return dataset.repeat().shuffle(buffer_size=1024).batch(batch_size)

def mixup_pairs(dataset):
    dataset = raw_batchs(dataset)
    for x, y in dataset:
        indexes = tf.random.shuffle(range(x.shape[0]))
        yield x, indexes, y, tf.gather(y, indexes)

def mix(model, mix_policy, x, indexes, lbda):
    if mix_policy == 'input':
        shuffled = tf.gather(x, indexes)
        # print(x.shape, shuffled.shape, (1.-lbda).shape, lbda.shape)
        mixed = lbda*x + (1.-lbda)*shuffled
        mixed = model(mixed)
    else:
        raise ValueError
    return mixed

def criterion(logits, y):
    return tf.nn.sparse_softmax_cross_entropy_with_logits(y, logits)

def mixup_score(model, dataset, mix_policy, alpha=2., num_batchs_max=128):
    losses = []
    for (x, indexes, y, yt), _ in zip(mixup_pairs(dataset), range(num_batchs_max)):
        lbda = np.random.beta(alpha, alpha, size=(y.shape[0],1))
        mixed = mix(model, mix_policy, x, indexes, lbda)
        loss = lbda*criterion(mixed, y) + (1.-lbda)*criterion(mixed, yt)
        losses.append(loss)
    return float(tf.math.reduce_mean(losses))

def lipschitz_naive(model, dataset):
    from deel.lip import evaluate_lip_const
    dataset = raw_batchs(dataset)
    scores = []
    for x, label in dataset:
        score = evaluate_lip_const(x, eps=1e-4)
        scores.append(score)
    return tf.math.reduce_mean(scores)

def complexity(model, dataset):
    avg_loss = mixup_score(model, dataset, mix_policy='input')
    return avg_loss