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


def get_laplacian(flatten_x, label, batch_size, batched_image_shape):
    with GradientTape(persistent=True, watch_accessed_variables=False) as laplacian:
        outer.watch(flatten_x)
        print('flatten_x', flatten_x.shape)
        with GradientTape(persistent=False, watch_accessed_variables=False) as grad:
            x = tf.reshape(flatten_x, batched_image_shape)  # going back to image space
            print('x', x.shape)
            inner.watch(x)
            y = model(x)
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(label, y)
            print('loss', loss)
            loss = tf.expand_dims(loss, axis=-1)
            print('loss', loss)
            per_image_gradients = grad.batch_jacobian(loss, x)[0]
        print('per_image_gradients', per_image_gradients.shape)
        per_image_gradients = tf.keras.backend.flatten(per_image_gradients)
        print('per_image_gradients', per_image_gradients.shape)
        flatten_hessian_diag = laplacian.batch_jacobian(per_image_gradients, flatten_x)  # here lies the lie
    print('flatten_hessian_diag', flatten_hessian_diag.shape)
    hessian_diag = tf.reshape(flatten_hessian_diag, (batch_size, -1))
    print('hessian_diag', hessian_diag.shape)
    laplacians = tf.reduce_sum(hessian_diag, axis=1)  # laplacian is Trace of Hessian
    print('laplacians', laplacians.shape)
    return tf.reduce_mean(laplacians)  # average over batch


def complexity(model, dataset):
    dummy_input = next(dataset.take(1).batch(1).__iter__())[0]  # warning: one image disappears
    output_shape = model(dummy_input).shape
    batch_size = 30
    batched_image_shape = dummy_input.shape
    num_batchs_max = 1
    measures = []
    dataset = raw_batchs(dataset, batch_size)
    for (x, label), _ in zip(dataset, progress_bar(num_batchs_max)):
        flatten_x = tf.tf.keras.backend.flatten(x)
        measure = get_laplacien(flatten_x, label, batch_size, batched_image_shape)
        measures.append(measure)
    return float(tf.reduce_mean(measures))

