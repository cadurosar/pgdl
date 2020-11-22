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
import scipy

def complexity(model, dataset):
	m = tf.keras.metrics.Mean()
	@tf.function()
	def compute_accuracy(inputs,labels):
		"""Get output from nn with respect to intermediate layers."""
		logits = model(inputs)
		ce = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
		labels=labels, logits=logits))
		m.update_state(ce)
	number_of_classes = 10
	for i, (x, y) in enumerate(dataset.batch(500,drop_remainder=True)):
		compute_accuracy(x,y)
	return m.result().numpy()
