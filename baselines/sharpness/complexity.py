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

# This complexity compute a specific notion of sharpness of a function.


import numpy as np
import tensorflow as tf

def complexity(model, ds):


  @tf.function
  def compute_bounds(weights, orig_weights, mag):
    max_perturb = [tf.abs(v) * mag for v in orig_weights]
    upper_bound = [v + vp for v, vp in zip(orig_weights, max_perturb)]
    lower_bound = [v - vp for v, vp in zip(orig_weights, max_perturb)]
    return upper_bound, lower_bound

  @tf.function
  def compute_clipped_weights(weights, upper_bound, lower_bound):
    clipped_weights = []
    for i, v in enumerate(weights):
      clipped_v = tf.minimum(tf.maximum(v, lower_bound[i]), upper_bound[i])
      clipped_weights.append(clipped_v)
    return clipped_weights

  # =================================
  @tf.function
  def optimize(loss_fn, optimizer, data):
    x, y = data
    y = tf.one_hot(y, 10)
    with tf.GradientTape() as tape:
      logits = model(x)
      loss = loss_fn(logits, y)
    variables = model.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))

  # =================================
  @tf.function
  def predict(x):
    logits = model(x)
    pred = tf.cast(tf.argmax(logits, axis=-1), tf.int32)
    return pred

  def get_noise(weights, scale):
    uniform_noise = []
    for w in weights:
      uniform_noise.append(np.random.uniform(low=-scale/2, high=scale/2, size=w.shape.as_list()))
      uniform_noise[-1] *= w
    return uniform_noise

  def estimate_accuracy(model, dataset, iteration):
    acc = 0.0
    for i in range(iteration):
      x, y = dataset.next()
      pred = predict(x).numpy()
      acc += np.mean(pred == y)
    return acc / iteration

  batched_ds = iter(ds.shuffle(1000).repeat(-1).batch(64))
  orig_weights = model.get_weights()
  cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
  optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3)

  h, l = 0.5, 0.000000
  target_deviate = 0.1
  epsilon = 1e-5
  target_accuracy = 0.9
  for i in range(20):
    m = (h + l) / 2.
    min_accuracy = 10.0
    for k in range(3):
      if min_accuracy < 0.8: continue
      model.set_weights(orig_weights)
      for j in range(20):
        data_pair = batched_ds.next()
        optimize(cce, optimizer, data_pair)
        curr_weights = [tf.convert_to_tensor(w) for w in model.get_weights()]
        upper, lower = compute_bounds(curr_weights, orig_weights, tf.convert_to_tensor(m))
        clipped_weights = compute_clipped_weights(curr_weights, upper, lower)
        model.set_weights(clipped_weights)
      estimate = []
      repeat = 3 if min_accuracy < 0.8 else 10
      estimate_mean = estimate_accuracy(model, batched_ds, repeat)
      min_accuracy = min(min_accuracy, estimate_mean)

    deviate = 1.0 - min_accuracy  # training accuracy - current accuracy
    if h - l < epsilon or abs(deviate - target_deviate) < 1e-2:
      break
    if deviate > target_deviate:
      h = m
    else:
      l = m
  return m

