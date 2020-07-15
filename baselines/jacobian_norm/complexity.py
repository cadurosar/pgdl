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
# Main contributor: Pierre Foret, July 2020

# This complexity computes the norm of the jacobian wrt to activations.

import numpy as np
import tensorflow as tf


def complexity(model, dataset):

  @tf.function()
  def get_jacobian(inputs):
      """Get jacobians with respect to intermediate layers."""
      with tf.GradientTape(persistent=True) as tape:
          out = model(inputs, tape=tape)
      dct = {}
      for i, l in enumerate(model.layers):
          try:
              dct[i] = tape.batch_jacobian(out, l._last_seen_input)
          except AttributeError:  # no _last_seen_input, layer not wrapped (ex: flatten)
              dct[i] = None
      return dct

  avg = tf.keras.metrics.Mean()
  for i, (x, y) in enumerate(dataset.batch(16)):
    J = get_jacobian(x)
    fro_norm = np.mean([np.mean(np.square(v)) for v in J.values()])
    avg.update_state(fro_norm)
    if i == 32:  # only 512 examples for efficiency
      break
  return avg.result().numpy()
