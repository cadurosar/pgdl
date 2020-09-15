"""Manifold Mixup Score.

python3 ingestion_program/ingestion_tqdm.py sample_data sample_result_submission ingestion_program manifold_mixup
python3 scoring_program/score.py sample_data sample_result_submission scores

python3 ingestion_program/ingestion_tqdm.py ../datasets/public_data sample_result_submission ingestion_program manifold_mixup
python3 scoring_program/score.py ../datasets/public_data sample_result_submission scores"""

import os
import numpy as np
import tensorflow as tf
try:
    import tqdm
    def progress_bar(num_batchs):
        return tqdm.tqdm(range(num_batchs), leave=False, ascii=True)
except:
    def progress_bar(num_batchs):
        return range(num_batchs)

def balanced_batchs(dataset, num_labels, batch_size=256):
    classes = []
    for label in range(num_labels):
        cur_class = dataset.filter(lambda data, y: tf.math.equal(y, label))
        cur_class = cur_class.repeat().shuffle(batch_size)
        classes.append(cur_class)
    return tf.data.experimental.sample_from_datasets(classes).batch(batch_size)

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

#######################################################################
############################### Lipschitz #############################
#######################################################################

@tf.function
def penalty(batch_squared_norm, one_lipschitz=False):
    batch_norm = tf.math.sqrt(batch_squared_norm)
    if one_lipschitz:
        return (batch_norm - 1.0) ** 2
    return batch_norm

@tf.function
def evaluate_lip(model, x, labels, softmax):
    with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(x)
        y = model(x)
        if softmax:
            y = tf.nn.softmax(y)
        y = tf.gather(y, indices=labels, axis=1)  # axis=0 is batch dimension, axis=1 is logits
    dy_dx = tape.gradient(y, x)
    batch_squared_norm = tf.math.reduce_sum(dy_dx ** 2, axis=list(range(1,len(dy_dx.shape))))
    grad_penalty = penalty(batch_squared_norm, one_lipschitz=True)
    lips = tf.math.reduce_mean(grad_penalty)
    return lips

def lipschitz_score(model, dataset, num_batchs_max, softmax):
    dataset = raw_batchs(dataset)
    scores = []
    progress = progress_bar(num_batchs_max)
    for (x, labels), _ in zip(dataset, progress):
        score = evaluate_lip(model, x, labels, softmax)
        scores.append(score)
    return float(tf.math.reduce_mean(scores))

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


#######################################################################
#################### Catastrophic forgetting ##########################
#######################################################################

def witness_set(model, dataset, num_batchs_witness):
    xs, ys = [], []
    progress = progress_bar(num_batchs_witness)
    for (x, _, _, _), _ in zip(dataset, progress):
        y = model(x)
        xs.append(x)
        ys.append(y)
    return xs, ys

def dumb_training(model, dataset, num_dumb_batchs):
    progress = progress_bar(num_dumb_batchs)
    opt = tf.keras.optimizers.Adam()
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    for (x, _, _, yt), _ in zip(dataset, progress):
        with tf.GradientTape() as tape:
            y = model(x)
            loss = loss_fn(yt, y)
        gradients = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(gradients, model.trainable_variables))

def witness_error(model, witness):
    l2_dists = []
    progress = progress_bar(len(witness[0]))
    for x, y, _ in zip(*witness, progress):
        logits = model(x)
        dist = tf.norm(y - logits, axis=1)  # logits axis, not batch
        l2_dists.append(dist)
    return float(tf.math.reduce_mean(l2_dists))

def catastrophic(model, dataset, num_batchs_witness, num_dumb_batchs):
    dataset = mixup_pairs(dataset)
    witness = witness_set(model, dataset, num_batchs_witness)
    dumb_training(model, dataset, num_dumb_batchs)
    loss = witness_error(model, witness)
    return loss

#######################################################################
################ Graph Distances Based Measure ########################
#######################################################################

#@tf.function
def partial_forward(model, x, layer_cut):
    for layer in model.layers[:layer_cut]:
        x = layer(x)
    return x

def resume_model(model, layer_cut):
    def f(x):
        for layer in model.layers[layer_cut:]:
            x = layer(x)
        return x
    return f

#@tf.function
def adj_matrix(batch):
    batch_dim, data_dim, dummy_dim = (batch.shape[0],), batch.shape[1:], (1,)
    batch_left = tf.reshape(batch, dummy_dim + batch_dim + data_dim)
    batch_right = tf.reshape(batch, batch_dim + dummy_dim + data_dim)
    delta = batch_left - batch_right
    norms = tf.math.reduce_sum(delta ** 2, axis=2)
    return norms

#@tf.function
def normalized(adj):
    diag = tf.linalg.diag_part(adj)
    diag = tf.linalg.diag(tf.math.rsqrt(diag))
    return diag @ adj @ diag

#@tf.function
def similarity(adj):
    return tf.math.exp(-adj)

#@tf.function
def thresholding(adj, num_edges_max):
    top_k, _ = tf.math.top_k(tf.reshape(-adj, [-1]), num_edges_max)
    threshold = -top_k[-1]
    mask = (adj <= threshold)
    return mask

#@tf.function
def get_graph(x, num_edges_max):
    adj = adj_matrix(x)
    adj = thresholding(adj, num_edges_max)
    return adj

#@tf.function
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

@tf.function
def jacob_lip(model, x):
    with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(x)
        y = model(x)
    dy_dx = tape.gradient(y, x)
    batch_squared_norm = tf.math.reduce_sum(dy_dx ** 2, axis=list(range(1,len(dy_dx.shape))))
    grad_penalty = penalty(batch_squared_norm, one_lipschitz=True)
    lips = tf.math.reduce_mean(grad_penalty)
    return lips

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


#######################################################################
############################# Complexity ##############################
#######################################################################

def complexity(model, dataset):
    # model.summary()
    public_data = False
    num_batchs_max = 8 if public_data else 24
    # avg_loss = lipschitz_interpolation(model, dataset, num_batchs_max, softmax=True, alpha=2.)
    # avg_loss = lipschitz_score(model, dataset, num_batchs_max, softmax=True)
    # avg_loss = mixup_score(model, dataset, num_batchs_max, mix_policy='input')
    # avg_loss = catastrophic(model, dataset, num_batchs_witness=num_batchs_max, num_dumb_batchs=4)
    avg_loss = graph_lip(model, dataset, num_batchs_max, almost_k_regular=8, layer_cut=-1, input_mixup=False)
    return avg_loss
