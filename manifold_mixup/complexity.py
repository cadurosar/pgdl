"""Manifold Mixup Score.

Command: python3 ingestion_program/ingestion_tqdm.py ../datasets/public_data sample_result_submission ingestion_program manifold_mixup

python scoring_program/score.py ../datasets/public_data sample_result_submission scores"""

import os
import numpy as np
import tensorflow as tf
try:
    # raise ValueError
    import tqdm
    tqdm_pb = True
except:
    tqdm_pb = False


def raw_batchs(dataset, batch_size=256):
    return dataset.repeat().shuffle(buffer_size=100).batch(batch_size)

def mixup_pairs(dataset):
    dataset = raw_batchs(dataset)
    for x, y in dataset:
        indexes = tf.random.shuffle(range(x.shape[0]))
        yield x, indexes, y, tf.gather(y, indexes)

def mix(model, mix_policy, x, indexes, lbda):
    z = tf.gather(x, indexes)
    if mix_policy == 'input':
        mixed = lbda*x + (1.-lbda)*z
        mixed = model(mixed)
        return mixed
    elif mix_policy == 'manifold':
        def forward(layer, *inputs):
            if len(inputs) == 1:
                return layer(inputs[0])
            return layer(inputs[0]), layer(inputs[1])
        inputs = x, z
        for layer in model.layers:
            inputs = forward(layer, *inputs)
            if layer.name == 'Flatten':
                inputs = lbda*inputs[0] + (1.-lbda)*inputs[1]
        return inputs
    raise ValueError

def criterion(logits, y):
    return tf.nn.sparse_softmax_cross_entropy_with_logits(y, logits)

def mixup_score(model, dataset, mix_policy, alpha=2., num_batchs_max=256):
    losses = []
    progress = tqdm.tqdm(range(num_batchs_max), leave=False) if tqdm_pb else range(num_batchs_max)
    for (x, indexes, y, yt), _ in zip(mixup_pairs(dataset), progress):
        shape = (y.shape[0],) + (1,)*(len(x.shape)-1)
        lbda = np.random.beta(alpha, alpha, size=shape)
        mixed = mix(model, mix_policy, x, indexes, lbda)
        loss = lbda*criterion(mixed, y) + (1.-lbda)*criterion(mixed, yt)
        losses.append(loss)
    return float(tf.math.reduce_mean(losses))

#######################################################################
############################### Lipschitz #############################
#######################################################################


def evaluate_lip_const(model, x, eps=1e-4, seed=None):
    y_pred = model.predict(x)
    # x = np.repeat(x, 100, 0)
    # y_pred = np.repeat(y_pred, 100, 0)
    x_var = x + K.random_uniform(
        shape=x.shape, minval=eps * 0.25, maxval=eps, seed=seed
    )
    y_pred_var = model.predict(x_var)
    dx = x - x_var
    dfx = y_pred - y_pred_var
    ndx = K.sum(K.square(dx), axis=range(1, len(x.shape)))
    ndfx = K.sum(K.square(dfx), axis=range(1, len(y_pred.shape)))
    lip_cst = K.sqrt(K.max(ndfx / ndx))
    print("lip cst: %.3f" % lip_cst)
    return lip_cst

def lipschitz_naive(model, dataset):
    from deel.lip import evaluate_lip_const
    dataset = raw_batchs(dataset)
    scores = []
    for x, label in dataset:
        score = evaluate_lip_const(x, eps=1e-4)
        scores.append(score)
    return tf.math.reduce_mean(scores)

def complexity(model, dataset):
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    avg_loss = mixup_score(model, dataset, mix_policy='manifold')
    return avg_loss