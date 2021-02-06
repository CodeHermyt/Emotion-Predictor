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

#emotion_model_info = emotion_model.fit_generator(
 #      train_generator,
  #      steps_per_epoch= 28709//64,
   #     epochs=50,
    #    validation_data=test_generator,
     #   validation_steps=7178//64
      #  )

#emotion_model.save_weights('model.h5')

emotion_model.load_weights('model.h5')

#initializing with cv2
cv2.ocl.setUseOpenCL(False)

emotion_dict = {0:"angery", 1:"Disgusted", 2:"fearful", 3:"Happy", 4:"neutral", 5:"sad",6:"surprised"}

cap = cv2.VideoCapture(0)
while(True):
    ret, frame = cap.read()
    
    if not ret:
        break
    bound_box = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    num_faces = bound_box.detectMultiScale(gray_frame,scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in num_faces:
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
        emotion_prediction = emotion_model.predict(cropped_img)
        maxindex = int(np.argmax(emotion_prediction))
        cv2.putText(frame, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('Video', cv2.resize(frame,(1200,860),interpolation = cv2.INTER_CUBIC))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break










