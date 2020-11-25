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

@tf.function()
def knn_tf(matrix,k):
	values, indices = tf.nn.top_k(matrix, k=k+1, sorted=True)
	thresholds = tf.reshape(values[:,-1],(-1,1))
	adjacence_matrix = tf.cast((matrix >= thresholds),tf.float32) # Create adjacence_matrix
	adjacence_matrix = tf.linalg.set_diag(adjacence_matrix, tf.zeros(adjacence_matrix.shape[0:-1]))
	adjacence_matrix = tf.clip_by_value(adjacence_matrix+tf.transpose(adjacence_matrix),0,1)
	return adjacence_matrix * matrix
	
@tf.function()
def get_distances(a, b):
    return tf.math.sqrt(tf.math.reduce_sum(tf.math.square(tf.expand_dims(a, axis=1) - tf.expand_dims(b, axis=0)),axis=2))

@tf.function()
def cosine(values):
    values = tf.math.l2_normalize(values,axis=1)
    values = tf.matmul(values,tf.transpose(values))
    values = tf.clip_by_value(values,0,1)
    return values

def generate_laplacian(values,k=20,use_mask=False,mask=None):
    values = tf.reshape(values,[values.shape[0],-1])
    matrix = cosine(values)#.numpy()
    if use_mask:
        matrix = matrix*mask
    adj = knn_tf(matrix,k)
    degree = tf.reduce_sum(adj,axis=1)
    degree = tf.linalg.diag(degree)
    laplacian = degree - adj
    return laplacian

@tf.function()
def smoothness(laplacian,targets):
    smoothness = tf.matmul(tf.transpose(targets),laplacian)
    smoothness = tf.matmul(smoothness,targets)
    smoothness = tf.linalg.trace(smoothness)
    return smoothness/(2*50*1000)



def complexity(model, dataset):
	@tf.function()
	def get_smoothness(inputs,labels):
		"""Get output from nn with respect to intermediate layers."""
		output = model(inputs)
		one_hot = tf.one_hot(labels,tf.reduce_max(labels)+1)
		output = smoothness(generate_laplacian(output),one_hot)
		outputs = list()
		for layer in model.layers[-2:]:
			outputs.append(smoothness(generate_laplacian(layer._last_seen_input),one_hot))
		outputs.append(output)
		return outputs
	values = list()
	for i, (x, y) in enumerate(dataset.batch(500,drop_remainder=True)):
		smoothnesses = get_smoothness(x,y)
		smoothness_rate = list()
		for ii in range(1,len(smoothnesses)):
			smoothness_rate.append(abs(smoothnesses[ii-1]-smoothnesses[ii]))
		values.append(np.mean(smoothness_rate))
		if i == 10:  # only 10000 examples for efficiency
			break
	return np.median(values)
