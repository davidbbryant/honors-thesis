import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image as img

 

fashion_mnist = tf.keras.datasets.fashion_mnist

 

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

 

print(train_images.shape)
print(test_images.shape)

 

class_names = ['T-shirt', 'Pants', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']

 

train_images = train_images /255.0  # scale these between 0 and 1 (8-bit color)
test_images = test_images / 255.0  # scale these between 0 and 1 (8-bit color)

 

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),   # convert from 2D matrix to 1D array
    tf.keras.layers.Dense(128, activation='relu'),  # fully connected network. Rectified Linear activation function
    tf.keras.layers.Dense(50, activation='relu'),   
    tf.keras.layers.Dense(10)  # Since there are 10 categories, I need 10 neurons/nodes
])

 

# ADAM (Adaptive Moment Estimation) instead of SGD as our learning function. SGD gets around 86/87% accuracy, 
# while ADAM gets around 91% accuracy

 

# Crossentropy calculates the differences between two probability distributions, see page 149. We want to minimize
# cross entropy so we use that as our loss function
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

 

model.fit(train_images, train_labels, epochs=10)

 

model.summary()

 

probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])

 

predictions = probability_model.predict(test_images)

 

print(class_names[np.argmax(predictions[1000])])

 

plt.figure()
plt.imshow(test_images[1000])
plt.colorbar()
plt.grid(False)
plt.show()