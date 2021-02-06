#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 20:52:42 2021

@author: ayush
"""


# importing Libraries

import numpy as np
import cv2

import tensorflow as tf

from keras.layers import Conv2D
from keras.preprocessing.image import ImageDataGenerator


# Initializing train and test generator

train_dir = 'data/train'
test_dir = 'data/test'

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        directory= train_dir,
        target_size = (48,48),
        batch_size = 64,
        color_mode = "grayscale",
        class_mode="categorical"
        )

test_generator = test_datagen.flow_from_directory(
        directory=test_dir,
        target_size = (48,48),
        batch_size = 64,
        color_mode= "grayscale",
        class_mode = "categorical"
        )


# Building the CNN architecture

emotion_model = tf.keras.models.Sequential()

emotion_model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape = (48,48,1)))
emotion_model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
emotion_model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=2))
emotion_model.add(tf.keras.layers.Dropout(0.25))

emotion_model.add(Conv2D(128, kernel_size=(3,3), activation='relu'))
emotion_model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=2))
emotion_model.add(Conv2D(128, kernel_size=(3,3), activation='relu'))
emotion_model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=2))
emotion_model.add(tf.keras.layers.Dropout(0.25))

emotion_model.add(tf.keras.layers.Flatten())
emotion_model.add(tf.keras.layers.Dense(1024, activation='relu'))
emotion_model.add(tf.keras.layers.Dropout(0.5))
emotion_model.add(tf.keras.layers.Dense(7, activation='softmax'))


# Compiling and training the model

from keras.optimizers import Adam     # optimizer for compiling

emotion_model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0001, epsilon=1e-6), metrics=['accuracy'])

emotion_model_info = emotion_model.fit_generator(
        train_generator,
        steps_per_epoch= 28709//64,
        epochs=50,
        validation_data=test_generator,
        validation_steps=7178//64
        )

emotion_model.save_weights('model.h5')

