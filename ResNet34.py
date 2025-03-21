import tensorflow as tf
from tensorflow import keras
from functools import partial
from tensorflow.keras.datasets import fashion_mnist
import numpy as np
from sklearn.model_selection import train_test_split

defaultConv2D = partial(keras.layers.Conv2D, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal', use_bias=False)


(x_trainf, y_trainf), (x_test, y_test) = fashion_mnist.load_data()

x_trainf = x_trainf.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

x_train, x_valid, y_train, y_valid = train_test_split(x_trainf, y_trainf, test_size=0.1)

# tf expects a (28, 28, 1) shape, so we add 
x_train = x_train[..., np.newaxis] 
x_valid = x_valid[..., np.newaxis]
x_test = x_test[..., np.newaxis]


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
        for layer in self.skip_layers:
            skip_Z = layer(skip_Z)
        
        return self.activation(Z+skip_Z)
        
resNet = keras.Sequential([
    defaultConv2D(64, kernel_size = 7, strides = 2, input_shape=[28,28,1]),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('relu'),
    keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same'),
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

etCallback = keras.callbacks.EarlyStopping(restore_best_weights = True, patience=5)
lrCallback = keras.callbacks.ReduceLROnPlateau(factor=0.5, monitor='val_loss', patience=3)

resNet.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
resNet.fit(x_train, y_train, epochs=20, validation_data=(x_valid, y_valid), callbacks=[etCallback, lrCallback])

loss, acc = resNet.evaluate(x_test, y_test)

x_new = x_test[:5]
y_pred = resNet.predict(x_new)

print('acc: ', acc*100,'%')
print('Classes: ',  y_test[:5])
print('Predicted: ' , np.argmax(y_pred, axis=1))