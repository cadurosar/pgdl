"""Manifold Mixup Score.

python3 ingestion_program/ingestion_tqdm.py sample_data sample_result_submission ingestion_program manifold_mixup
python3 scoring_program/score.py sample_data sample_result_submission scores

python3 ingestion_program/ingestion_tqdm.py ../datasets/public_data sample_result_submission ingestion_program adversarial
python3 scoring_program/score.py ../datasets/public_data sample_result_submission scores"""

import os
import numpy as np
import tensorflow as tf


@tf.function
def projected_gradient(model, x, label, num_steps, num_steps, epsilon, inf_dataset=0., sup_dataset=1.):
    label = tf.broadcast_to(label, shape=[x.shape[0], 1])
    non_batch_axis = list(range(1,len(x.shape)))
    for i in range(num_steps):
        y = f(x)
        loss = tf.nn.softmax_cross_entropy_with_logits(label, y)
        g = tf.gradients(loss, x)
        x = x + g
        x = tf.clip_by_norm(x, epsilon)  # return to epsilon ball
        x = tf.clip_by_value(x, inf_dataset, sup_dataset)  # return to image manifold


def adversarial_score(model, dataset, num_batchs_max):
    pass

def complexity(model, dataset):
    # model.summary()
    num_batchs_max = 32  # 4096
    avg_loss = adversarial_score(model, dataset, num_batchs_max)
    return avg_loss


