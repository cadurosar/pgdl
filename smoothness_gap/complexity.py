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

def knn_over_matrix(matrix,k):
    temp = np.argsort(-matrix,axis=1)[:,k] # Get K biggest index
    thresholds = matrix[np.arange(matrix.shape[0]),temp].reshape(-1,1) # Transform matrix into a column matrix of maximums
    adjacence_matrix = (matrix >= thresholds)*1.0 # Create adjacence_matrix
    np.fill_diagonal(adjacence_matrix, 0)
    weighted_adjacence_matrix = adjacence_matrix * matrix # Create weigthed adjacence_matrix
    return weighted_adjacence_matrix

def get_distances(a, b):
    return tf.math.sqrt(tf.math.reduce_sum(tf.math.square(tf.expand_dims(a, axis=1) - tf.expand_dims(b, axis=0)),axis=2))


def generate_laplacian(values,k=5):
    values = tf.reshape(values,[values.shape[0],-1])
    matrix = get_distances(values,values)
    matrix = tf.math.exp(-2*matrix).numpy()
    adj = knn_over_matrix(matrix,k)
#    adj = matrix
    degree = np.sum(adj,axis=1)
    degree[degree<=0] = 1
    degree = np.power(degree,-1/2)
    degree = np.diag(degree)
    new_adj = np.matmul(degree,np.matmul(adj,degree))
    laplacian = np.eye(new_adj.shape[0]) - new_adj
#    laplacian = degree - adj
    return laplacian

def smoothness(laplacian,targets):
    smoothness = np.matmul(targets.T,laplacian)
    smoothness = np.matmul(smoothness,targets)
    smoothness = np.trace(smoothness)
    if smoothness < 1e-4:
        smoothness = 0
    return smoothness/(2*50*1000)


def complexity(model, dataset):
	@tf.function()
	def get_output(inputs):
		"""Get output from nn with respect to intermediate layers."""
		last_layer = model(inputs)
		penultimate_layer = model.layers[-1]._last_seen_input
		return last_layer,penultimate_layer
	values = list()
	for i, (x, y) in enumerate(dataset.batch(100)):
		last,penultimate = get_output(x)
		n_values = np.max(y) + 1
		one_hot = np.eye(n_values)[y]
		lap_last = generate_laplacian(last)
		lap_penultimate = generate_laplacian(penultimate)
		smoothness_last = smoothness(lap_last,one_hot)
		smoothness_penultimate = smoothness(lap_penultimate,one_hot)
		smoothness_gap = np.square(smoothness_last - smoothness_penultimate)#/max(smoothness_last,smoothness_penultimate)
		values.append(smoothness_gap)
		if i == 100:  # only 3000 examples for efficiency
			break
	return np.mean(values)
