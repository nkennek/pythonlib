#!/usr/bin/env python
# -- coding:utf-8 --

import tensorflow as tf
import numpy as np
from auto_encoder import AutoEncoder

class Layer(object):
    def __init__(self, in_dim, out_dim, function):
        self.W = tf.Variable(np.random.uniform(low=-0.08, high=0.08, size=(in_dim, out_dim)).astype('float32'))
        self.b = tf.Variable(np.zeros([out_dim]).astype('float32'))
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.function = function
        self.params = [self.W, self.b]

    def f_prop(self, x):
        u = tf.matmul(x, self.W) + self.b
        self.z = self.function(u)
        return self.z

#全結合層
class Dense(Layer):
    def __init__(self, in_dim, out_dim, function):
        super().__init__(in_dim, out_dim, function)

#全結合層(AutoEncoderを含む)
class DenseSDA(Layer):
    def __init__(self, in_dim, out_dim, function):
        super().__init__(in_dim, out_dim, function)
        self.ae = AutoEncoder(in_dim, out_dim, self.W, self.b, self.function)

    def pretrain(self, x, noise):
          cost, reconst_x = self.ae.reconst_error(x, noise)
          return cost, reconst_x
