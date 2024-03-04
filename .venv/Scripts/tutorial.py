import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import pathlib
data_dir = "D:\OneDrive\Files\School\HNRS Thesis\Data\Categories\Finished"

batch_size = 30
# Base size is 1024
img_size = 64

train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  labels="inferred",
  class_names=None, #["Circle", "Cross", "Diagonal Line", "Grouped Mass", "L or Rectangular", "Radiating Line", "S or Compound Curve"],
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_size, img_size),
  batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_size, img_size),
  batch_size=batch_size)

AUTOTUNE = tf.data.AUTOTUNE

# New locale
class_names = train_ds.class_names

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

normalization_layer = tf.keras.Sequential([
    layers.Resizing(img_size, img_size),

    # Augmentation for data
    layers.RandomFlip("horizontal",
                       input_shape=(img_size,
                                    img_size,
                                    3)),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1)
])


normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
# Notice the pixel values are now in `[0,1]`.
print(np.min(first_image), np.max(first_image))

# class_names decleration was here


num_classes = len(class_names)

model = Sequential([
  layers.Rescaling(1./255, input_shape=(img_size, img_size, 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),

  # Adding dropout
  layers.Dropout(0.2),

  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Originally 10 epochs
# Set back to 50 epochs
epochs=50
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(10, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.ylim(bottom=0, top=1.05)
plt.xlabel('Number of Epochs')
plt.ylabel('Prediction Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.ylim(bottom=-0.1, top=5.1)
plt.xlabel('Number of Epochs')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

#Added to test on outside images
data_dir = "D:\OneDrive\Files\School\HNRS Thesis\Data\Test Images"
#data_dir = "D:\OneDrive\Files\School\HNRS Thesis\Data\Categories\Finished"

batch_size = 1
# Base size is 1024
img_size = 64

test_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  image_size=(img_size, img_size),
  batch_size=batch_size)

#sunflower_url = "D:\OneDrive\Files\School\HNRS Thesis\Data\Test Images\landscape-painting.jpg"
#sunflower_path = tf.keras.utils.get_file('test_image', origin=sunflower_url)

#img = tf.keras.utils.load_img(
    #sunflower_path, target_size=(img_height, img_width)
#)
#img_array = tf.keras.utils.img_to_array(img)
#img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(test_ds)
score = tf.nn.softmax(predictions[0])

#test = list(data_dir.glob('Testing/*'))
#PIL.Image.open(str(test[0]))
#print(np.argmax(predictions[0]))

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)