"""Adversarial score.

python3 ingestion_program/ingestion_tqdm.py sample_data sample_result_submission ingestion_program laplacian
python3 scoring_program/score.py sample_data sample_result_submission scores

python3 ingestion_program/ingestion_tqdm.py ../datasets/public_data sample_result_submission ingestion_program laplacian
python3 scoring_program/score.py ../datasets/public_data sample_result_submission scores"""


import os
import math
from functools import wraps
from time import time
import numpy as np
import tensorflow as tf
from utils import progress_bar, balanced_batchs, raw_batchs
import tqdm


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print(f'func:{f.__name__} took: {te-ts:2.4f} sec')
        return result
    return wrap


# @timing
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

# @timing
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

@tf.function
def hutchinson_forward(model, x, label, unbatched_image_shape):
    v = tf.random.uniform(x.shape, -1., 1.)
    v = tf.math.sign(v)  # v ~ Rademacher
    with tf.autodiff.ForwardAccumulator(x,v) as acc:
        with tf.GradientTape() as tape:
            tape.watch(x)
            y = model(x)
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(label, y)
            loss = tf.reduce_mean(loss)
        backward = tape.gradient(loss, x)
    Hv = acc.jvp(backward)
    laplacian = tf.reduce_sum(v * Hv)
    return laplacian

def median_of_means(datas, num_splits):
    datas = tf.split(datas, num_splits)  # Median of Means
    datas = [tf.reduce_mean(data) for data in datas]
    datas = np.median([data.numpy() for data in datas])
    return datas

# @timing
def hutchinson_trick(model, x, label, batch_size, monte_carlo_samples, unbatched_image_shape):
    laplacians = []
    for _ in range(monte_carlo_samples):
        laplacian = hutchinson_forward(model, x, label, unbatched_image_shape)
        laplacians.append(laplacian)
    # print(tf.stack(laplacians), tf.reduce_mean(laplacians), tf.math.reduce_std(laplacians))
    return median_of_means(laplacians, 5)

def complexity(model, dataset):
    dummy_input = next(dataset.take(1).batch(1).__iter__())[0]  # warning: one image disappears
    batch_size = 12  # 8 works
    method = 'hutchinson'
    monte_carlo_samples = 15
    unbatched_image_shape = tuple(dummy_input.shape[1:])
    batched_image_shape = (batch_size,) + unbatched_image_shape
    num_examples = 1300
    num_batchs_max = num_examples // batch_size
    dataset = raw_batchs(dataset, batch_size)
    progress = tqdm.tqdm(range(num_examples), leave=False, ascii=True)
    deltas = []
    jacobs = []
    hutchinsons = []
    verbose = False
    for (x, label), _ in zip(dataset, range(num_batchs_max)):
        if 'jacob' in method:
            jacob = get_laplacian_jacob_based(model, x, label, batch_size, batched_image_shape)
            jacobs.append(jacob)
            if verbose:            
                print('jacob', jacob)
                print('jacobs', tf.reduce_mean(jacobs))
        if 'hessian' in method:
            hessian = get_laplacian_hessian_based(model, x, label, batch_size)
            if verbose:
                print('hessian', hessian)
        if 'hutchinson' in method:
            hutchinson = hutchinson_trick(model, x, label, batch_size, monte_carlo_samples, unbatched_image_shape)
            hutchinsons.append(hutchinson)
            if verbose:
                print('hutchinson', hutchinson)
                print('hutchinsons', tf.reduce_mean(hutchinsons))
        if 'hutchinson' in method and 'jacob' in method:
            deltas.append((jacob - hutchinson)**2.) 
            if verbose:
                print('delta', tf.reduce_mean(deltas)**0.5)
        progress.update(batch_size)
    return float(tf.reduce_mean(hutchinsons))

