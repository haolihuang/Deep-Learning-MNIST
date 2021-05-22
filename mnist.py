# -*- coding: utf-8 -*-
"""
Author: Hao-Li Huang
Exercise from XSEDE Big Data and Machine Learning Workshop, Apr 2021
"""

import tensorflow as tf

# Load built-in MNIST data set
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Reshape, convert to float, normalize
train_images = train_images.reshape(60000, 28, 28, 1).astype('float32') / 255
test_images = test_images.reshape(10000, 28, 28, 1).astype('float32') / 255

# Build neural network
# 2x Conv2D layers to get 2D feature maps
# MaxPooling2D to reduce dimension and reduce overfitting
# Followed by 2x 1D Dense layers 
# 2x Dropout layers are added to reduce overfitting
model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2), 
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, activation='softmax')
        ])

model.compile(optimizer='adam', # stochastic gradient descent with momentum
              loss='sparse_categorical_crossentropy', # good with softmax, penalizes bad answers
              metrics=['accuracy'])

# Save data to log_dir to be read by TensorBoard
# It creates a folder named TB_logDir in the current directory
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir='TB_logDir',
    histogram_freq=1)

# Train neural network and save data to history
history = model.fit(train_images, train_labels,
          batch_size=128, # number of samples per gradient update
          epochs=30, # go through the entire data x times
          verbose=1,
          validation_data=(test_images, test_labels),
          callbacks=[tensorboard_callback])


# Plot train/test accuracy
import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Test')
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.show()



