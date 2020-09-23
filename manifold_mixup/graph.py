import tensorflow as tf
from utils import *


#######################################################################
################ Graph Distances Based Measure ########################
#######################################################################

@tf.function
def adj_matrix(batch):
    batch_dim, data_dim, dummy_dim = (batch.shape[0],), batch.shape[1:], (1,)
    batch_left = tf.reshape(batch, dummy_dim + batch_dim + data_dim)
    batch_right = tf.reshape(batch, batch_dim + dummy_dim + data_dim)
    delta = batch_left - batch_right
    squared_norms = tf.math.reduce_sum(delta ** 2, axis=list(range(2,len(delta.shape))))
    return squared_norms

@tf.function
def normalized(adj):
    diag = tf.linalg.diag_part(adj)
    diag = tf.linalg.diag(tf.math.rsqrt(diag))
    return diag @ adj @ diag

@tf.function
def similarity(adj):
    return tf.math.exp(-adj)

@tf.function
def thresholding(adj, num_edges_max):
    top_k, _ = tf.math.top_k(tf.reshape(-adj, [-1]), num_edges_max)
    threshold = -top_k[-1]
    mask = (adj <= threshold)
    return mask

@tf.function
def get_graph(x, num_edges_max):
    adj = adj_matrix(x)
    adj = thresholding(adj, num_edges_max)
    return adj

@tf.function
def edge_interpolation(mask, x, labels, num_points=3):
    data_dim = tuple(x.shape[1:])
    pairs = tf.where(mask)  # retrieve
    left, right = pairs[:,0], pairs[:,1]
    x_left, x_right = tf.gather(x, left), tf.gather(x, right)
    x_left, x_right = tf.expand_dims(x_left, axis=0), tf.expand_dims(x_right, axis=0)
    lbdas = tf.range(0., 1., 1/(num_points-1), dtype=tf.float32)
    lbdas = tf.reshape(lbdas, (-1,1)+(1,)*len(data_dim))
    z = lbdas * x_left + (1. - lbdas) * x_right  # broadcast
    z = tf.reshape(z, shape=(-1,)+data_dim)  # flatten to batch dim 1
    return z, tf.gather(labels, left), tf.gather(labels, right)

def graph_lip(model, dataset, num_batchs_max, almost_k_regular, layer_cut, input_mixup):
    output_shape = model(next(dataset.take(1).batch(1).__iter__())[0]).shape
    num_labels = int(output_shape[-1])
    dataset = balanced_batchs(dataset, num_labels)
    lips = []
    resumed = resume_model(model, layer_cut=layer_cut) if not input_mixup else model
    progress = progress_bar(num_batchs_max)
    for (x, labels), _ in zip(dataset, progress):
        latent = partial_forward(model, x, layer_cut=layer_cut)
        num_edges_max = latent.shape[0] * almost_k_regular
        mask = get_graph(latent, num_edges_max)
        if input_mixup:
            interpolated, _, _ = edge_interpolation(mask, x, labels, num_points=5)
        else:
            interpolated, _, _ = edge_interpolation(mask, latent, labels, num_points=5)
        lip = jacob_lip(resumed, interpolated)
        lips.append(lip)
    return float(tf.math.reduce_mean(lip))