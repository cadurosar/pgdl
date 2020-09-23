import tensorflow as tf
from utils import *

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