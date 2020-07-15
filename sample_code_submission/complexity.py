"""Sample complexity measure that measures the l2 norms of the weights."""

import numpy as np
import tensorflow as tf


def complexity(model, dataset):
    weights = model.get_weights()
    norm = sum([np.linalg.norm(w)**2 for w in weights])
    return norm

