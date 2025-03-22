import tensorflow_datasets as tfds
import tensorflow as tf

dataset, info = tfds.load("tf_flowers", as_supervised=True, with_info=True)
dataset_size = info.splits["train"].num_examples
class_names = info.features['label'].names
n_classes = info.features['label'].num_classes

test_set_raw, valid_set_raw, train_set_raw = tfds.load(
"tf_flowers",
split=["train[:10%]", "train[10%:25%]", "train[25%:]"],
as_supervised=True)

batch_size = 32
preprocess = tf.keras.Sequential([
    tf.keras.layers.Resizing(height=224, width=224, crop_to_aspect_ratio=True),
    tf.keras.layers.Lambda(tf.keras.applications.xception.preprocess_input)
])

train_set = train_set_raw.map(lambda X, y: (preprocess(X), y))
train_set = train_set.shuffle(1000, seed=42).batch(batch_size).prefetch(1)

test_set = test_set_raw.map(lambda X, y: (preprocess(X), y))
test_set = test_set.shuffle(1000, seed=42).batch(batch_size)

valid_set = valid_set_raw.map(lambda X, y: (preprocess(X), y))
valid_set = valid_set.shuffle(1000, seed=42).batch(batch_size)

base_model = tf.keras.applications.xception.Xception(wieghts='imagenet', include_top=False)

for layer in base_model.layers:
    layer.trainable = False

avg = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
class_output = tf.keras.layers.Dense(n_classes, activation='softmax')
loc_output = tf.keras.layers.Dense(4)(avg)

model = tf.keras.Model(inputs=base_model.inputs, outputs=[loc_output, class_output])

model.compile(loss=['mse', 'sparse_categorical_crossentropy'], loss_weights=[0.2,0.8], optimizer='SGD', metrics=['accuracy'])
