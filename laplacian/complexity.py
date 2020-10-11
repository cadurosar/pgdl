"""Adversarial score.

python3 ingestion_program/ingestion_tqdm.py sample_data sample_result_submission ingestion_program laplacian
python3 scoring_program/score.py sample_data sample_result_submission scores

python3 ingestion_program/ingestion_tqdm.py ../datasets/public_data sample_result_submission ingestion_program laplacian
python3 scoring_program/score.py ../datasets/public_data sample_result_submission scores"""

import os
import math
import numpy as np
import tensorflow as tf
from utils import progress_bar, balanced_batchs, raw_batchs
import tqdm


def get_laplacian_jacob_based(model, x, label, batch_size, batched_image_shape):
    flatten_x = tf.reshape(x, (-1, 1))
    with tf.GradientTape(persistent=True, watch_accessed_variables=False) as laplacian:
        laplacian.watch(flatten_x)
        with tf.GradientTape(persistent=False, watch_accessed_variables=False) as grad:
            x = tf.reshape(flatten_x, batched_image_shape)  # going back to image space
            grad.watch(x)
            y = model(x)
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(label, y)
            loss = tf.expand_dims(loss, axis=-1)
        per_image_gradients = grad.batch_jacobian(loss, x)
        per_image_gradients = tf.reshape(per_image_gradients, (-1, 1))
    flatten_hessian_diag = laplacian.batch_jacobian(per_image_gradients, flatten_x)  # here lies the lie
    hessian_diag = tf.reshape(flatten_hessian_diag, (batch_size, -1))
    laplacians = tf.reduce_sum(hessian_diag, axis=1)  # laplacian is Trace of Hessian
    return tf.reduce_mean(laplacians)  # average over batch

@tf.function
def get_laplacian_hessian_based(model, x, label, batch_size):
    y            = model(x)
    loss         = tf.nn.sparse_softmax_cross_entropy_with_logits(label, y)
    loss         = tf.reduce_mean(loss)
    hessian      = tf.hessians(loss, x)[0]
    hessian      = tf.reshape(hessian, (tf.size(x),tf.size(x)))
    diag_hessian = tf.linalg.diag_part(hessian)
    laplacian    = tf.reduce_sum(diag_hessian)
    return laplacian

def hutchinson_trick(model, x, label, batch_size, monte_carlo_samples, unbatched_image_shape):
    laplacians = []
    for _ in range(monte_carlo_samples):
        v = tf.random.uniform((1,)+unbatched_image_shape, -1., 1.)
        v = tf.math.sign(v)  # v ~ Rademacher
        v = tf.broadcast_to(v, x.shape)
        with tf.autodiff.ForwardAccumulator(x,v) as acc:
            with tf.GradientTape() as tape:
                tape.watch(x)
                y = model(x)
                loss = tf.nn.sparse_softmax_cross_entropy_with_logits(label, y)
                loss = tf.reduce_mean(loss)
            backward = tape.gradient(loss, x)
        Hv = acc.jvp(backward)
        laplacian = tf.reduce_sum(v * Hv)
        laplacians.append(laplacian)
    return tf.reduce_mean(laplacians)


def complexity(model, dataset):
    dummy_input = next(dataset.take(1).batch(1).__iter__())[0]  # warning: one image disappears
    batch_size = 8
    method = 'jacob'
    monte_carlo_samples = 32
    unbatched_image_shape = tuple(dummy_input.shape[1:])
    batched_image_shape = (batch_size,) + unbatched_image_shape
    num_examples = 256
    num_batchs_max = num_examples // batch_size
    measures = []
    dataset = raw_batchs(dataset, batch_size)
    progress = tqdm.tqdm(range(num_examples), leave=False, ascii=True)
    for (x, label), _ in zip(dataset, range(num_batchs_max)):
        if method == 'jacob':
            measure = get_laplacian_jacob_based(model, x, label, batch_size, batched_image_shape)
        elif method == 'hessian':
            measure = get_laplacian_hessian_based(model, x, label, batch_size)
        elif method == 'hutchinson':
            measure = hutchinson_trick(model, x, label, batch_size, monte_carlo_samples, unbatched_image_shape)
        measures.append(measure)
        progress.update(batch_size)
            
    return float(tf.reduce_mean(measures))

