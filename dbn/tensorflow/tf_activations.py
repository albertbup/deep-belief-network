#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
name:    relusigmoid for tensorflow
purpose: activation function (for neural networks)
@author: OvG
'''

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
#import os
# remove warnings from tf
#os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

def relusigmoid(x):
    '''forward pass'''
    return tf.multiply(tf.cast(x>0,tf.float32),(2.0 / (1.0 + tf.math.exp(-x)) - 1))

def d_relusigmoid(x):
    '''forward pass'''
    return tf.multiply(tf.cast(x>0,tf.float32),2.0*(x * (1.0 - x)))

def tf_relusigmoid(x, name=None):
    '''forward pass for tf'''
    with ops.name_scope(name, "relusigmoid",[x]) as name:
        y = py_func(func=relusigmoid,
                        inp=[x],
                        Tout=[tf.float32],
                        name=name,
                        grad=relusigmoidgrad)  # call the gradient
        return y[0]

def tf_d_relusigmoid(x,name=None):
    '''backward pass for tf'''
    with ops.name_scope(name, "d_relusigmoid",[x]) as name:
        y = tf.py_function(func=d_relusigmoid,
                        inp=[x],
                        Tout=[tf.float32],
                        name=name)#stateful=False
        return y[0]

def py_func(func, inp, Tout, stateful=True, name=None, grad=None):
    '''get gradients from graph'''
    # Need to generate a unique name to avoid duplicates:
    rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 1E+8))
    tf.RegisterGradient(rnd_name)(grad)  
    g = tf.get_default_graph()
    with g.gradient_override_map({"PyFunc": rnd_name}):

        return tf.py_function(func=func, inp=inp, Tout=Tout, name=name)#stateful=False

def relusigmoidgrad(op, grad):
    '''gradient function'''
    x = op.inputs[0]
    n_gr = tf_d_relusigmoid(x)
    return grad * n_gr
