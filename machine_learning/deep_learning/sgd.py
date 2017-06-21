#!/usr/bin/env python
# -- coding:utf-8 --

import tensorflow as tf
import numpy as np

def sgd(cost, params, params_buf = None, eps = np.float32(0.01), momentum = 0.9, decay = 1.0e-7):
    g_params = tf.gradients(cost, params)
    updates = []
    if params_buf is None:
        for param, g_param in zip(params, g_params):
            if g_param != None:
                #updates.append(param.assign(param - eps*g_param))
                updates.append(tf.assign(param, param - eps*g_param))
            return updates
    else:
        m = momentum
        for param, param_buf, g_param in zip(params, g_params):
            if g_param != None:
                updates.append(param.assign(param - eps*g_param + eps*m*param_buf - decay*param))
                updates.append(param_buf.assign(param))
    return updates
