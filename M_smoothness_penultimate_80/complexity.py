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
#    adjacence_matrix = tf.clip_by_value(adjacence_matrix+tf.transpose(adjacence_matrix),0,1)
    adjacence_matrix = tf.clip_by_value(adjacence_matrix,0,1)
    return adjacence_matrix #* matrix
    
@tf.function()
def get_distances(A):
    r = tf.reduce_sum(A*A, 1)
    # turn r into column vector
    r = tf.reshape(r, [-1, 1])
    D = r - 2*tf.matmul(A, A,transpose_b=True) + tf.transpose(r)
    return D

@tf.function()
def RBF(values,gamma=1):
    distances = get_distances(values)
    std = tf.math.reduce_std(distances)
    gamma = 1/(3*tf.math.abs(std))
    values = tf.exp(-gamma*distances)
    return values

def RBF_knn(values,k,gamma=1):
    distances = -get_distances(values)
    distances = -knn_tf(distances,k)
    std = tf.math.reduce_std(distances)
    gamma = 1/(3*tf.math.abs(std))
    values = tf.exp(-gamma*distances)
    return values


@tf.function()
def cosine(values):
    values = tf.matmul(values,values,transpose_b=True)
    values = tf.clip_by_value(values,0,1)
    return values

@tf.function()
def generate_laplacian(values,k=int(1),use_mask=False,mask=None):
    values = tf.reshape(values,[values.shape[0],-1])
    values = tf.math.l2_normalize(values,axis=1)
    matrix = RBF(values)#.numpy()
#    adj = RBF_knn(values,k)#.numpy()
#    matrix = cosine(values)#.numpy()
    if use_mask:
        matrix = matrix*mask
    adj = knn_tf(matrix,k)
#    adj[adj > 0] = 1
#    adj = adj+tf.eye(500)
#    adj = matrix
    degree = tf.reduce_sum(adj,axis=1)
    degree = tf.math.pow(degree,-0.5)
    degree = tf.linalg.diag(degree)
    laplacian = tf.matmul(degree,tf.matmul(adj,degree))
#    laplacian = tf.matmul(laplacian,laplacian)
#    laplacian = degree - adj
    laplacian = tf.eye(500) - laplacian
    return laplacian

@tf.function()
def smoothness(laplacian,targets):
    smoothness = tf.matmul(tf.transpose(targets),laplacian)
    smoothness = tf.matmul(smoothness,targets)
    smoothness = tf.linalg.trace(smoothness)
    smoothness = tf.clip_by_value(smoothness/(2*2*500),0,1000)
    return smoothness

@tf.function()
def log10(x):
  numerator = tf.math.log(x)
  denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
  return numerator / denominator

@tf.function()
def error(laplacian,signal):
    noisy = tf.matmul(laplacian,signal)
#    result = tf.cast(tf.argmax(noisy,1),tf.int32)
#    labels = tf.cast(tf.argmax(signal,1),tf.int32)
#    error = 100*(1-tf.math.reduce_mean(tf.cast(tf.math.equal(result,labels),tf.float32)))
#    print(noisy[0])
#    print(result[:10])
#    print(labels[:10])
#    print(accuracy)
    error = 20*log10(tf.norm(signal)/tf.norm(noisy))
#    error = tf.clip_by_value(smoothness/(2*50*1000),0,1000)
    return error

#@tf.function()
def mixup_output(model, x, y, one_hot, mix_policy, alpha=0):
    indexes = tf.random.shuffle(range(x.shape[0]))
    if alpha > 0:
        #lbda = np.random.normal(0.5,0.1,size=(y.shape[0],1)).astype(np.float32)
        lbda = np.random.beta(alpha, alpha, size=(y.shape[0],1)).astype(np.float32)
    else:
        lbda = np.ones((y.shape[0],1)).astype(np.float32)
    outputs = mix(model, mix_policy, x, indexes, lbda.reshape(-1,1,1,1))
    shuffled_hot = tf.gather(one_hot, indexes)
    one_hot = lbda*one_hot + (1.-lbda)*shuffled_hot
    #one_hot2 = lbda*one_hot + (1.-lbda)*shuffled_hot
    #one_hot = tf.concat([one_hot,one_hot2],0)
    return outputs,one_hot

@tf.function()
def mix(model, mix_policy, x, indexes, lbda):
    outputs = list()
    if mix_policy == 'input':
        shuffled = tf.gather(x, indexes)
        # print(x.shape, shuffled.shape, (1.-lbda).shape, lbda.shape)
        mixed = lbda*x + (1.-lbda)*shuffled
#        output = tf.concat([x,mixed],0)
        output = model(mixed)
    elif mix_policy == 'manifold':
        length = len(model.layers)
        middle_index = length//2
        first_half = model.layers[:middle_index]
        output = x
        for layer in first_half:
            output = layer(output)
#            outputs.append(output)
        second_half = model.layers[middle_index:]
        
        shuffled = tf.gather(output, indexes)
        # print(x.shape, shuffled.shape, (1.-lbda).shape, lbda.shape)
        output = lbda*output + (1.-lbda)*shuffled
        
        for layer in second_half:
            output = layer(output)
#            outputs.append(output)
        mixed = output
    else:
        raise ValueError
    for layer in model.layers:
        outputs.append(layer._last_seen_input)
    outputs.append(output)
    return outputs


def complexity(model, dataset):
    @tf.function()
    def get_smoothness(inputs,labels):
        """Get output from nn with respect to intermediate layers."""
#       output = model(inputs)
        one_hot = tf.one_hot(labels,tf.reduce_max(labels)+1)
        results, one_hot = mixup_output(model, inputs, labels, one_hot, "input",2)
        outputs = list()
        layers_to_get = (len(model.layers)//3)+1
#       layers_to_get = 3
#        results = model.layers
        for layer_output in [results[-2]]:
            value = smoothness(generate_laplacian(layer_output),one_hot)
            outputs.append(value)
#        outputs.append(smoothness(generate_laplacian(model.layers[-1]._last_seen_input),one_hot))
#        outputs.append(smoothness(generate_laplacian(results[-2]),one_hot))
#        outputs.append(smoothness(generate_laplacian(results[-2]),one_hot))
        return outputs
    number_of_classes = 10 #All tasks for the moment have 10 classes
    examples_per_class = 50
#   for x,y in dataset.batch(100):
#       number_of_classes = max(number_of_classes,np.max(y.numpy())+1)
    values = list()
    datasets = list()
    
    for a in range(number_of_classes):
        datasets.append(dataset.filter(lambda data, labels: tf.equal(labels, a)).repeat().shuffle(100).batch(examples_per_class).__iter__()) 

    for kk in range(80): # number of graphs
        x = list()
        y = list()
        for data in datasets:
            x_, y_ = data.next()
            x.append(x_)
            y.append(y_)
        x = tf.reshape(tf.convert_to_tensor(x),[-1,*tf.shape(x)[2:]])
        y = tf.reshape(tf.convert_to_tensor(y),[-1])
#       noise = tf.random_normal(shape=tf.shape(x), mean=0.0, stddev=0.1, dtype=tf.float32) 

        smoothnesses = get_smoothness(x,y)
        smoothness_rate = list()
#        for ii in range(1,len(smoothnesses)):
#            values.append(tf.math.abs(smoothnesses[ii-1]-smoothnesses[ii]))
        values.append(tf.math.reduce_min(smoothnesses))
    return np.median(values)