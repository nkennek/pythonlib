#!/usr/bin/env python
# -- coding:utf-8 --

import tensorflow as tf
import numpy as np

class AutoEncoder(object):
    def __init__(self, vis_dim, hid_dim, W, b, function = lambda x:x):
        self.W = W
        self.b = b
        self.a = tf.Variable(np.zeros(vis_dim).astype('float32'), name = 'a')
        self.function = function
        self.params = [self.W, self.a, self.b]

    def encode(self, x):
        u = tf.matmul(x, self.W) * self.b
        return self.function(u)

    def decode(self, y):
        u = tf.matmul(y, tf.transpose(self.W)) + self.a
        return self.function(u)

    def f_prop(self, x):
        y = self.encode(x)
        return self.decode(y)

    def reconst_error(self, x, noise):
        tlide_x = tf.multiply(x, noise)
        reconst_x = self.f_prop(tlide_x)
        error = -tf.reduce_mean(tf.reduce_sum(x * tf.log(tf.clip_by_value(reconst_x, 1e-10, 1.0)) + (1. - x) * tf.log(1. - tf.clip_by_value(reconst_x, 1e-10, 1.0)), axis = 1))
        return error, reconst_x

if __name__ == "__main__":
    #MNISTを分類
    from tf_layers import DenseSDA
    from sgd import sgd

    from sklearn.utils import shuffle
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import f1_score
    from tensorflow.examples.tutorials.mnist import input_data

    rng = np.random.RandomState(1234)
    random_state = 42

    #データ読み込み
    mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
    mnist_X, mnist_y = mnist.train.images, mnist.train.labels
    train_X, valid_X, train_y, valid_y = train_test_split(mnist_X, mnist_y, test_size=0.1, random_state=random_state)

    layers = [
        DenseSDA(784, 500, tf.nn.tanh),
        DenseSDA(500, 200, tf.nn.tanh),
        DenseSDA(200, 50, tf.nn.tanh),
        DenseSDA(50, 10, tf.nn.softmax)
    ]

    def f_props(layers, x):
        params = []
        for layer in layers:
            x = layer.f_prop(x)
            params += layer.params
        return x, params

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        #事前学習
        for l, layer in enumerate(layers[:-1]):
            corruption_level = np.float(0.3)
            batch_size = 100
            n_batches = train_X.shape[0]// batch_size
            n_epochs = 10

            x = tf.placeholder(tf.float32)
            noise = tf.placeholder(tf.float32)

            cost, reconst_x = layer.pretrain(x, noise)
            params = layer.params
            grad = tf.gradients(cost, params)
            print(grad)
            updates = sgd(cost, params)
            train = tf.group(*updates)
            for epoch in range(n_epochs):
                X = shuffle(train_X, random_state=random_state)
                err_all = []
                for i in range(n_batches):
                    start = i * batch_size
                    end = start + batch_size

                    _noise = rng.binomial(size=X[start:end].shape, n=1, p=1-corruption_level)
                    _, err = sess.run([train, cost], feed_dict={x: X[start:end], noise: _noise})
                    #print(sess.run(grad, feed_dict={x: X[start:end], noise: _noise}))
                    err_all.append(err)
                print('Pretraining:: layer: %d, Epoch: %d, Error: %lf' % (l+1, epoch+1, np.mean(err_all)))
        #グラフ構築
        x = tf.placeholder(tf.float32, [None, 784])
        t = tf.placeholder(tf.float32, [None, 10])

        y, params_learn = f_props(layers, x)
        cost_learn = -tf.reduce_mean(tf.reduce_sum(t * tf.log(tf.clip_by_value(y, 1e-10, 1.0)), 1))
        updates_learn = sgd(cost_learn, params_lean)
        train = tf.group(*updates_learn)
        valid = tf.argmax(y, 1)

        #学習
        n_epochs = 100
        batch_size = 10
        n_batches = train_X.shape[0] // batch_size

        for epoch in range(n_epochs):
            train_X, train_y = shuffle(train_X, train_y, random_state=random_state)
            for i in range(n_batches):
                start = i * batch_size
                end = start + batch_size
                sess.run(train, feed_dict={x: train_X[start:end], t: train_y[start:end]})
                pred_y, valid_cost = sess.run([valid, cost], feed_dict={x: valid_X, t: valid_y})
                print('EPOCH: %i, Validation cost: %.3f Validation F1: %.3f' % (epoch + 1, valid_cost, f1_score(np.argmax(valid_y, 1).astype('int32'), pred_y, average='macro')))
