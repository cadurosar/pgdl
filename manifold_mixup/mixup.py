import tensorflow as tf
import numpy as np
from utils import *

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

@tf.function
def criterion(logits, y):
    return tf.nn.sparse_softmax_cross_entropy_with_logits(y, logits)

def mixup_score(model, dataset, num_batchs_max, mix_policy, alpha=2.):
    losses = []
    mix_fn = mix_input if mix_policy == 'input' else mix_manifold
    progress = progress_bar(num_batchs_max)
    for (x, indexes, y, yt), _ in zip(mixup_pairs(dataset), progress):
        shape = (y.shape[0],) + (1,)*(len(x.shape)-1)
        lbda = tf.constant(np.random.beta(alpha, alpha, size=shape), dtype=tf.float32)
        mixed = mix_fn(model, x, indexes, lbda)
        loss = lbda*criterion(mixed, y) + (1.0-lbda)*criterion(mixed, yt)
        loss = tf.math.reduce_mean(loss)
        losses.append(loss)
    return float(tf.math.reduce_mean(losses))

def lipschitz_interpolation(model, dataset, num_batchs_max, softmax, alpha=2.):
    scores = []
    progress = progress_bar(num_batchs_max)
    for (x, indexes, labels, _), _ in zip(mixup_pairs(dataset), progress):
        shape = (labels.shape[0],) + (1,)*(len(x.shape)-1)
        lbda = tf.constant(np.random.beta(alpha, alpha, size=shape), dtype=tf.float32)
        z = tf.gather(x, indexes)
        mixed = lbda*x + (1.0-lbda)*z
        score = evaluate_lip(model, mixed, labels, softmax)
        scores.append(score)
    return float(tf.math.reduce_mean(scores))
