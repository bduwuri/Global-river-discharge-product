# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 18:35:56 2023

@author: duvvuri.b
"""
import tensorflow as tf
import keras
import random
keras.utils.set_random_seed(42)
tf.random.set_seed(42)
random.seed(42)
# tf.compat.v1.enable_eager_execution()
from keras.models import Sequential
from keras.layers import Dense,Dropout,Input,Flatten
from scikeras.wrappers import KerasRegressor, KerasClassifier
# import tensorflow as tf
# import keras.backend as K
import keras
import sys  # DONT DELETE 
import numpy as np
# from functools import partial
# from tensorflow.python.ops import math_ops
# from tensorflow.python.framework import dtypes
# from tensorflow.python.framework import ops
from world_map_functions import FindLayerNodesLinear

class models:

    def custom_loss(self,y_true, y_pred):
        # print(y_pred,"__________________________ytp")
        loss_a = tf.keras.losses.mean_absolute_error(y_true[:,0], y_pred[:,0])
        loss_b = tf.keras.losses.mean_absolute_error(y_true[:,1], y_pred[:,1])*  2**tf.abs(y_true[:,1])
        # print(loss_a +loss_b,"loss_A")
        return loss_a+loss_b# +tf.reduce_mean(loss_b)

    def model_reg(self,n_layers, first_layer_nodes, last_layer_nodes, learning_rate, optimizer, dr, dr1,  activation_func):
        
        ann = Sequential() # Initialising ANN
        
        n_nodes = FindLayerNodesLinear(n_layers, first_layer_nodes, last_layer_nodes)

        for i in range(n_layers-1): # Adding multiple Hidden Layer
            if i==0:
                # ann.add(Flatten(input_dim= 50)) # Adding First Hidden Layer
                ann.add(Dense(n_nodes[i], activation='linear', input_dim=31))#kernel_regularizer='l1_l2',bias_regularizer='l1_l2'
                ann.add(Dropout(dr))
            else:
                ann.add(Dense(n_nodes[i], activation=activation_func))
                ann.add(Dropout(dr1))
        ann.add(Dense(units = last_layer_nodes,activation = 'linear'))
        
        opt = tf.keras.optimizers.get(optimizer)
        opt.learning_rate = learning_rate
        ##############################################    
        ann.compile(optimizer = opt, loss= 'mean_squared_error',
                        # metrics=['mean_squared_error']
                        )   
        return ann 
    
    def model_class(self,n_layers, first_layer_nodes, last_layer_nodes, learning_rate, optimizer, dr, dr1,  activation_func):
        
        
        ann = Sequential() # Initialising ANN
        
        n_nodes = FindLayerNodesLinear(n_layers, first_layer_nodes, last_layer_nodes)
        # ann.add(Flatten(input_dim= 50)) 
        # ann.add(Dense(units = 50, activation='linear'))
        # print(n_nodes)
        for i in range(n_layers-1): # Adding multiple Hidden Layer
            if i==0:
                # ann.add(Flatten(input_dim= 50)) # Adding First Hidden Layer
                ann.add(Dense(n_nodes[i], activation='linear', input_dim=36))#kernel_regularizer='l1_l2',bias_regularizer='l1_l2'
                ann.add(Dropout(dr))
            else:
                ann.add(Dense(n_nodes[i], activation=activation_func))
                ann.add(Dropout(dr1))
        # ann.add(Dense(units = n_nodes[-i],activation='relu'))
        ann.add(Dense(units = last_layer_nodes,activation = 'relu')) # Adding Output Layer
        
        ###############################################
        # Add optimizer with learning rate
        opt = tf.keras.optimizers.get(optimizer)
        opt.learning_rate = learning_rate
        
        ##############################################    
        ann.compile(optimizer = opt,\
                    loss='binary_crossentropy',
                        metrics=['accuracy'])
        
        return ann 