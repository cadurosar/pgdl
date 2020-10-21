"""Influence score.

python3 ingestion_program/ingestion_tqdm.py datasets/sample_data sample_result_submission ingestion_program influence
python3 scoring_program/score.py datasets/sample_data sample_result_submission scores

python3 ingestion_program/ingestion_tqdm.py datasets/public_data sample_result_submission ingestion_program influence
python3 scoring_program/score.py datasets/public_data sample_result_submission scores"""

import os
import math
from functools import wraps
from time import time
import numpy as np
import tensorflow as tf
from utils import progress_bar, balanced_batchs, raw_batchs
import tqdm



def get_weights_last_layers(model, n_last_layers: int, only_kernel: bool):
    """Get the weights of the n_last_layers.

    Args:
        model: Sequential Keras model
        n_last_layers: integer, number of last layers to take into account

    Return:
        the weights of the layers

    Remark:
        one might want to modify this function to only select kernel and not bias, for example
    """
    weights = []
    for layer in model.layers[n_last_layers:]:
        if only_kernel:
            weights.append(layer.weights[-1])
        else:
            weights += layer.weights

    return weights


@tf.function
def ce_loss(label, y):
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(label, y))


def flatten_grads(grads):
    """Return list of gradients flattened (line of Hessian)

    Args:
        grads: list of gradients of loss wrt each weight

    Returns:
        list flattened of gradients
    """
    return [tf.reshape(grad, -1) for grad in grads]


def get_hessian_wrt_parameters(model, weights, x_latent, n_last_layers, label):
    """Hessians of the model for input.

    Args:
        model: Sequential Keras model
        weights: list of weights belonging to the model

    Return:
        list of lists of second order derivatives

    Explanations:
        Assume there is n weights in weights, from weights[0] to weights[n-1]
        grads[i] contains the derivatives of loss wtr to weights[i]
        hessians[i][j] contains the derivatives of grads[i] wrt to weights[j]
        hessians[i][j].shape = weights[i].shape + weights[j].shape
        Yeah, for real, isn't that cool ?

    See:
        consider approximating the hessian as outer product of gradient with itself
        (rough approximation) but may be faster to compute and inverse the one using
        Jacobian, HOWEVER take care that outer product of gradient with itself is a
        rank one matrix, and hence is not invertible.
    """
    with tf.GradientTape(persistent=True) as tape_hess:
        with tf.GradientTape(persistent=True) as tape_grad:
            y = forward_from(model, x_latent, n_last_layers)
            loss = ce_loss(label, y)
            grads = tape_grad.gradients(loss, weights)  # list of gradients for each weight
        hessians = [tape_hess.jacobian(grad, weights) for grad in grads]

    return hessians


def merge_hessians(hessians, batch_hessians, step):
    """Merge a batch of Hessians into another batch of hessians.

    Args:
        hessians: list of lists of second order derivatives
        batch_hessians: list of lists of second order derivatives
        step: current step

    Return:
        list of lists of second order derivatives

    Warning: may lead to inaccuracies with moving average, but should be ok in average.
    """
    if hessians is None:
        return batch_hessians
    assert len(hessians) == len(batch_hessians)
    accumulated = []
    for outer_hes, outer_batch_hes in zip(hessians, batch_hessians):
        assert len(outer_hes) == len(outer_batch_hes)
        accumulated.append([])
        for hes, hes_batch in zip(outer_hes, outer_batch_hes):
            accumulated[-1].append(hes + (hes_batch - hes) / (step + 1))  # moving average

    return accumulated


def square_hessians(hessians, weights):
    """Return the hessians as big square matrix.

    Args:
        hessians: a list of lists of second order derivatives
        weights: a list of weights

    Return:
        the true Hessian (square matrix) of the model

    Remarks:
        Remember that hessians[i][j].shape = weights[i].shape + weights[j].shape

    This functions perform a reshaping so that hessians_flattened.shape = (n, n)
    with n = sum(product(weight.shape[i]) for all i) (THIS IS HUGE)
    """
    hessians_flattened = []
    for i in range(len(hessians)):
        hessians_i = hessians[i]
        weights_i_shape = tuple(weights[i].shape)
        hessians_i = [tf.reshape(hessians_i, shape=weights_i_shape + (-1,))]
        hessians_i = tf.concat(hessians_i, axis=-1)
        hessians_i = tf.reshape(hessians_i, shape=(-1, int(hessians_i.shape[-1])))
        hessians_flattened.append(hessians_i)
    hessians_flattened = tf.concat(hessians_flattened, axis=0)

    return hessians_flattened


def naive_exact_inverse_hessians(hessians, weights):
    """Return the inverse of the average of the hessians.

    Args:
        hessians: a list of lists of second order derivatives
        weights: a list of weights

    Return:
        the true inverse of Hessian (square matrix) of the model

    Remark: using conjugate gradient descent, the inverse of hessian can be computed more efficiently

    See:
        tf.linalg.experimental.conjugate_gradient for speedup
    """
    square = square_hessians(hessians, weights)
    square_inv = tf.linalg.inv(square)

    return square_inv


def get_avg_hessian(model, dataset, weights, num_batchs_max: int, n_last_layers: int):
    """Return the average of hessians.

    Args:
        model: a Sequential Keras model
        dataset: Tensorflow Dataset
        num_batchs_max: number of batchs to consider
        n_last_layer: number of layers to consider in the model (the last ones for examples)

    Return:
        a list of lists of second order derivatives, averaged over many batchs
    """
    progress = progress_bar(num_batchs_max)
    hessians = None  # will be updated after first iteration
    for (x, label), step in zip(dataset, progress):
        x_latent = forward_until(model, x, n_last_layers)  # no need to record in tape this forward, to save memory
        hessians_batch = get_hessian_wrt_parameters(model, weights, x, n_last_layers, label)
        hessians = merge_hessians(hessians, hessians_batch, step)
    return hessians


def dummy_forward(model, dataset):
    """Consume an example to initialize network and retrieve number of classes."""
    dummy_input = next(dataset.take(1).batch(1).__iter__())[0]  # warning: one image disappears
    output_shape = model(dummy_input).shape
    num_labels = int(output_shape[-1])

    return num_labels


def get_batch_gradients(model, label, x, weights):
    with tf.GradientTape() as tape:
        y = model(x)
        loss = ce_loss(label, y)  # average over batch
        grads = tape.gradients(loss, weights)
    grads = tf.concat(flatten_grads(grads))

    return grads


def naive_influence_avg(model, dataset, inv_hessian, weights, num_batchs_max):
    influences = []
    progress = progress_bar(num_batchs_max)
    for (x, label), _ in zip(dataset, progress):
        grads = get_batch_gradients(model, label, x, weights)
        influence_fn = inv_hessian @ grads
        influence_norm = tf.norm(influence_fn)
        influences.append(influence_norm)
    return float(tf.reduce_mean(influences))


def fast_cgd(model, dataset, hessian, weights, num_batchs_max):
    gd_fn = tf.linalg.experimental.conjugate_gradient
    progress = progress_bar(num_batchs_max)
    influences = []
    hessian_op = tf.linalg.LinearOperatorFullMatrix(hessian,
                                                    is_non_singular=True,  # WARNING
                                                    is_self_adjoint=True,
                                                    is_positive_definite=True,  # WARNING
                                                    is_square=True)
    for (x, label), _ in zip(dataset, progress):
        grads = get_batch_gradients(model, label, x, weights)
        named_tuple = gd_fn(hessian_op, grads, tol=1e-3, max_iter=20)
        influence_fn = named_tuple.x  # solution to problem
        influence_norm = tf.norm(influence_fn)
        influences.append(influence_norm)
    return float(tf.reduce_mean(influences))


def normalize_influence(influence, weights):
    """Normalize influence by square root of parameter space dimension.

    Args:
        influence: average influence norm
        weights: parameter space, as a list of weight Tensors

    Return:
        the influence divided by the square root of the parameter space dimension

    The goal of this function is to ensure comparable ranges within different networks.
    """
    param_dimensions = sum([float(tf.size(weight)) for weight in weights])

    return influence / math.sqrt(param_dimensions)


def complexity(model, dataset):
    num_labels = dummy_forward(model, dataset)
    num_batchs_hessian_estimation = 64
    num_batchs_influence          = 64
    batch_size                    = 16
    n_last_layers                 = -1  # last kernel layer only
    only_kernel                   = True
    conjugate_gd                  = False
    dataset         = raw_batchs(dataset, batch_size=batch_size)
    weights         = get_weights_last_layers(model, n_last_layers=n_last_layers, only_kernel=only_kernel)
    hessians        = get_avg_hessian(model, dataset, num_batchs_max=num_batchs_max, n_last_layers=n_last_layers)
    if conjugate_gd:
        hessian         = square_hessians(hessians)
        influence       = fast_cgd(model, dataset, hessian, weights, num_batchs_max)
    else:
        inv_hessian     = naive_exact_inverse_hessians(hessians, weights)
        influence       = naive_influence_avg(model, dataset, inv_hessian, weights, num_batchs_max)
    influence = normalized_influence(influence, weights)
    return influence
