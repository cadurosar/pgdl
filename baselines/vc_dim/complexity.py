# Copyright 2020 The PGDL Competition organizers.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Utilities for loading models for PGDL competition at NeurIPS 2020
# Main contributor: Yiding Jiang, July 2020

# This complexity compute an upper bound on the VC-dimension of a convnet.


import numpy as np
import tensorflow as tf

def complexity(model, ds):
  model_weights = model.get_weights()
  model_variables = model.trainable_variables
  kernels = []
  for v, w in zip(model_variables, model_weights):
    if 'kernel' in v.name:
      kernels.append(v)
  d = len(kernels)
  log_term = np.log(22*d*32**2)
  summation = 0
  for v in kernels:
    shape = v.get_shape().as_list()
    summation += np.prod(shape)
  return d + 2 * d * log_term * summation

