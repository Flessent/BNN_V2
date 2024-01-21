from larq.layers import QuantDense
import keras.backend as K
import tensorflow as tf
from tensorflow.keras import layers, constraints, initializers, models
import numpy as np
from tensorflow.keras.initializers import RandomNormal

from activations import*
from tensorflow.keras.models import Sequential

import larq as lq
from keras.layers import Dense, Activation, BatchNormalization,Dropout,InputLayer,Lambda

class Binarizer(layers.Layer):
    def __init__(self, **kwargs):
        super(Binarizer, self).__init__(**kwargs)

    def binarize(self, inputs):
        return tf.where(inputs < 0, -1.0, 1.0)

    def call(self, inputs):
        return self.binarize(inputs)

def describe_network(model):
    for layer in model.layers:
        if isinstance(layer, QuantDense):
            print(f'for LinearLayer weights are  : {layer.get_weights()[0]} shape : {len(layer.get_weights()[0])}')
        
        else:
            print(f'Unknown layer type: {type(layer)}')


class BNN(models.Sequential):
    def __init__(self,input_dim, output_dim, num_neuron_in_hidden_dense_layer=3, num_neuron_output_layer=3 ):
            super(BNN, self).__init__()
            kwargs = dict(input_quantizer="ste_sign",
              kernel_quantizer="ste_sign",
              kernel_constraint="weight_clip",
              use_bias=False)


            self.add(InputLayer(input_shape=(input_dim,)))
            self.add(QuantDense(num_neuron_in_hidden_dense_layer, input_shape=(input_dim,),  kernel_quantizer="ste_sign",
                          kernel_constraint="weight_clip",
                          use_bias=False))
            self.add(BatchNormalization(momentum=0.999, scale=False))
            #self.add(tf.keras.layers.Activation("relu"))
            self.add(Dropout(0.2))
            self.add(QuantDense(num_neuron_in_hidden_dense_layer, **kwargs))
            self.add(BatchNormalization(momentum=0.999, scale=False))
            self.add(Dropout(0.2))
            self.add(QuantDense(num_neuron_in_hidden_dense_layer, **kwargs))
            self.add(BatchNormalization(momentum=0.999, scale=False))          
            self.add(Dropout(0.2))
            self.add(QuantDense(num_neuron_in_hidden_dense_layer, activation='linear', use_bias=True))
            self.add(BatchNormalization(momentum=0.999, scale=False))
            #self.add(tf.keras.layers.Activation("relu"))
            #self.add(Lambda(lambda x: tf.sign))
            self.add(Dropout(0.3))
            #self.add(Activation("ste_sign"))
            self.add(QuantDense(num_neuron_output_layer, **kwargs ) )
            self.add(tf.keras.layers.Activation('sigmoid'))