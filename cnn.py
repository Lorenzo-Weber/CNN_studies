# In order to run the code you must have the following packages installed:
# - scikit-learn (pip install scikit-learn)
# - tensorflow (pip install tensorflow or pip install tensorflow[gpu])
# - PIL (pip install --upgrade Pillow)

from sklearn.datasets import load_sample_images
import tensorflow as tf

images = load_sample_images()['images']
images = tf.keras.layers.CenterCrop(height=70, width=120)(images)
images = tf.keras.layers.Rescaling(scale=1/255.0)(images)

# The kernel size defines the size of the receptive field
conv_layer = tf.keras.layers.Conv2D(filters=32, kernel_size=7, padding='same')
fmaps = conv_layer(images)

pool_layer = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2)

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
