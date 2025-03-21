import tensorflow as tf
from sklearn.datasets import load_sample_images

model = tf.keras.applications.ResNet50(weights='imagenet')

images = load_sample_images()['images']
images_resized = tf.keras.layers.Resizing(height=224, width=224, crop_to_aspect_ratio=True)(images)

inputs = tf.keras.applications.resnet50.preprocess_input(images_resized)

Y_proba = model.predict(inputs)

top_K = tf.keras.applications.resnet50.decode_predictions(Y_proba, top=3)
for image_index in range(len(images)):
    print(f"Image #{image_index}")
    for class_id, name, y_proba in top_K[image_index]:
        print(f" {class_id} - {name:12s} {y_proba:.2%}")    