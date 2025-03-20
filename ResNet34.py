import tensorflow as tf
from tensorflow import keras
from functools import partial

defaultConv2D = partial(keras.layers.Conv2D, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal', use_bias=False)

# Adds a residual input to the output 

class ResidualUnit(tf.keras.layers.Layer):
    def __init__(self, filters, strides=1, activation='relu', **kwargs):
        super().__init__(**kwargs)
        self.activation = keras.activations.get(activation)
        self.main_layers = [
            defaultConv2D(filters, strides=strides),
            keras.layers.BatchNormalization(),
            self.activation,
            defaultConv2D(filters),
            keras.layers.BatchNormalization()
        ]

        self.skip_layers = []

        if strides > 1:
            self.skip_layers = [
                    defaultConv2D(filters, kernel_size=1, strides=strides),
                    keras.layers.BatchNormalization()
            ]
    
    def call(self, inputs):
        Z = inputs

        for layer in self.main_layers:
            Z = layer(Z)
        
        skip_Z = inputs
        for lyaer in self.skip_layers:
            skip_Z = layer(skip_Z)
        
        return self.activation(Z+skip_Z)
        
resNet = keras.Sequential([
    defaultConv2D(64, kernel_size = 7, strides = 2, input_shape=[224,224,3]),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('relu'),
    keras.layers.MaxPool2D(pools_size=3, strides=2, padding='same'),
])

prev_filters = 64

# basically, in the resNet34 we have 3 Residual Units containing 64 filters, and so on
# but we need to set the stride to 1 if the residual units have the same filter amount
# else we set the stride to 2
for filters in [64] * 3 + [128] * 4 + [256] * 6 + [512] * 3:
    strides = 1 if filters == prev_filters else 2
    resNet.add(ResidualUnit(filters, strides=strides))
    prev_filters = filters

resNet.add(keras.layers.GlobalAvgPool2D())
resNet.add(keras.layers.Flatten())
resNet.add(keras.layers.Dense(10, activation='softmax'))

