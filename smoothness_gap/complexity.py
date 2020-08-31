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
	values, indices = tf.nn.top_k(matrix, k=tf.cast(k+1,tf.int32), sorted=True)
	thresholds = tf.reshape(values[:,-1],(-1,1))
	adjacence_matrix = tf.cast((matrix >= thresholds),tf.float32) # Create adjacence_matrix
	adjacence_matrix = tf.linalg.set_diag(adjacence_matrix, tf.zeros(adjacence_matrix.shape[0:-1]))
	adjacence_matrix = tf.clip_by_value(adjacence_matrix+tf.transpose(adjacence_matrix),0,1)
	return adjacence_matrix * matrix
	
@tf.function()
def get_distances(A):
	r = tf.reduce_sum(A*A, 1)
	# turn r into column vector
	r = tf.reshape(r, [-1, 1])
	D = r - 2*tf.matmul(A, A,transpose_b=True) + tf.transpose(r)
	return D

@tf.function()
def RBF(values,gamma=1):
    values = tf.exp(-gamma*get_distances(values))
    return values


@tf.function()
def cosine(values):
    values = tf.matmul(values,values,transpose_b=True)
    values = tf.clip_by_value(values,0,1)
    return values

@tf.function()
def generate_laplacian(values,k=int(50),use_mask=False,mask=None):
    values = tf.reshape(values,[values.shape[0],-1])
    values = tf.math.l2_normalize(values,axis=1)
    matrix = RBF(values)#.numpy()
#    matrix = cosine(values)#.numpy()
    if use_mask:
        matrix = matrix*mask
    adj = knn_tf(matrix,k)
#    adj = matrix
    degree = tf.reduce_sum(adj,axis=1)
#    degree = tf.math.pow(degree,-0.5)
    degree = tf.linalg.diag(degree)
#    laplacian = tf.matmul(degree,tf.matmul(adj,degree))
    laplacian = degree - adj
#    laplacian = tf.eye(500) - laplacian
    return laplacian

@tf.function()
def smoothness(laplacian,targets):
    smoothness = tf.matmul(tf.transpose(targets),laplacian)
    smoothness = tf.matmul(smoothness,targets)
    smoothness = tf.linalg.trace(smoothness)
    smoothness = tf.clip_by_value(smoothness/(2*50*1000),0,1000)
    return smoothness



def complexity(model, dataset):
	@tf.function()
	def get_smoothness(inputs,labels):
		"""Get output from nn with respect to intermediate layers."""
		output = model(inputs)
		one_hot = tf.one_hot(labels,tf.reduce_max(labels)+1)
		output = smoothness(generate_laplacian(output),one_hot)
		outputs = list()
		for layer in [model.layers[-1]]:
			outputs.append(smoothness(generate_laplacian(layer._last_seen_input),one_hot))
		outputs.append(output)
		return outputs
	number_of_classes = 10 #All tasks for the moment have 10 classes
	examples_per_class = 50
#	for x,y in dataset.batch(100):
#		number_of_classes = max(number_of_classes,np.max(y.numpy())+1)
	values = list()
	datasets = list()
	for a in range(number_of_classes):
		datasets.append(dataset.filter(lambda data, labels: tf.equal(labels, a)).batch(examples_per_class).__iter__()) 

	for kk in range(20): # number of graphs
		x = list()
		y = list()
		for data in datasets:
			x_, y_ = data.next()
			x.append(x_)
			y.append(y_)
		x = tf.reshape(tf.convert_to_tensor(x),[-1,*tf.shape(x)[2:]])
		y = tf.reshape(tf.convert_to_tensor(y),[-1])

		smoothnesses = get_smoothness(x,y)
		smoothness_rate = list()
		for ii in range(1,len(smoothnesses)):
			smoothness_rate.append(tf.math.abs(smoothnesses[ii-1]-smoothnesses[ii]))
		values.append(tf.math.reduce_mean(smoothness_rate))
#		values.append(smoothnesses[0])
	return np.median(values)
