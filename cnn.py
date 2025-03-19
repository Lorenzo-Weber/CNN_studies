# In order to run the code you must have the following packages installed:
# - scikit-learn (pip install scikit-learn)
# - tensorflow (pip install tensorflow or pip install tensorflow[gpu])
# - PIL (pip install --upgrade Pillow)

import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from sklearn.model_selection import train_test_split
import numpy as np

(x_trainf, y_trainf), (x_test, y_test) = fashion_mnist.load_data()

x_trainf = x_trainf.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

x_train, x_valid, y_train, y_valid = train_test_split(x_trainf, y_trainf, test_size=0.1)

# tf expects a (28, 28, 1) shape, so we add 
x_train = x_train[..., np.newaxis] 
x_valid = x_valid[..., np.newaxis]
x_test = x_test[..., np.newaxis]

# The kernel size defines the size of the receptive field
conv_layer = tf.keras.layers.Conv2D(filters=32, kernel_size=7, padding='same')

pool_layer = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2)

# Basically gets the max within the filters instead of going through all the filters
class DepthPool(tf.keras.layers.Layer):
    def __init__(self, pool_size,**kwargs ):
        super().__init__(**kwargs)
        self.pool_size = pool_size

    def call(self, inputs):
        shape = tf.shape(inputs)
        groups = shape[-1] // self.pool_size
        new_shape = tf.concat([shape[:-1], [groups, self.pool_size]], axis=0)

        return tf.reduce_max(tf.reshape(inputs, new_shape), axis=-1)

global_pool = tf.keras.layers.GlobalAvgPool2D()

# Creating a CNN
from functools import partial

DefaultConv2D = partial(tf.keras.layers.Conv2D, kernel_size=3, padding='same', activation='relu', kernel_initializer='he_normal')

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=[28,28,1]),
    DefaultConv2D(filters=64, kernel_size=7),
    tf.keras.layers.MaxPool2D(),
    DefaultConv2D(filters=128),
    DefaultConv2D(filters=128),
    tf.keras.layers.MaxPool2D(),
    DefaultConv2D(filters=256),
    DefaultConv2D(filters=256),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=128, activation='relu', kernel_initializer='he_normal'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(units=64, activation='relu', kernel_initializer='he_normal'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(units=10, activation='softmax')
])

alexNet = tf.keras.Sequential([
    tf.keras.layers.Input(shape=[28,28,1]),
    DefaultConv2D(filters=96, kernel_size=11),
    tf.keras.layers.MaxPool2D(),
    DefaultConv2D(filters=256, kernel_size=5),
    tf.keras.layers.MaxPool2D(),
    DefaultConv2D(filters=384, kernel_size=3),
    DefaultConv2D(filters=256, kernel_size=3),
    DefaultConv2D(filters=256, kernel_size=3),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=4096, activation='relu', kernel_initializer='he_normal'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(units=4096, activation='relu', kernel_initializer='he_normal'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(units=10, activation='softmax'),
])

et_callback = tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
lr_callback = tf.keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5, monitor='val_loss')

model.compile(loss='sparse_categorical_crossentropy', optimizer='nadam', metrics=['accuracy'])

model.fit(x_train, y_train, validation_data=(x_valid, y_valid) ,epochs=10, callbacks=[et_callback, lr_callback])

loss, acc = model.evaluate(x_test, y_test)

x_new = x_test[:3]
y_pred = model.predict(x_new)

import numpy as np
print('real values: ', y_test[:3])
print('predicted: ', np.argmax(y_pred, axis=1))
print('accuracy: ', acc*100, '%')