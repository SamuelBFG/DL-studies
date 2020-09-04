import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Just disables the warning, doesn't enable AVX/FMA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# print(tf.__version__)

fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
print(train_images.shape)

# plt.imshow(train_images[9])
# plt.show()

train_images = train_images/255.0
test_images = test_images/255.0

model = tf.keras.Sequential([
	tf.keras.layers.Flatten(input_shape=(28,28)),
	tf.keras.layers.Dense(128, activation=tf.nn.relu),
	tf.keras.layers.Dense(10, activation=tf.nn.softmax)
	])

model.compile(optimizer=tf.optimizers.Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5)

print(model.evaluate(test_images, test_labels))