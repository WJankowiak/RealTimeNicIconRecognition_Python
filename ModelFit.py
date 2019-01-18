# -*- coding: utf-8 -*-
"""
@author: Wojciech Jankowiak
"""

import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D,Dropout,MaxPooling2D,Flatten
from keras.utils import to_categorical
import ReadImages as RI

PATH_TO_FILE =  'E:/STUDIA/'
IMAGES_DIRECTORY = 'E:/Studia/RealTimeSignRecognition/signs'
DATASET_SIZE = 6930

ys, xs = RI.load_data_from_coords(IMAGES_DIRECTORY)   

xs = np.reshape(xs, (DATASET_SIZE, 16384))
xs = xs.reshape(xs.shape[0], 128, 128, 1)

ys = to_categorical(ys, 14)
print(xs.shape[0], 'train samples')


model = Sequential()
model.add(Conv2D(32, kernel_size=(3,3),
                 activation='relu',
                 input_shape=[128,128,1]))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(14, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.sgd(),
              metrics=['accuracy'])
print(xs.shape[0], 'train samples')
model.fit(xs,ys ,
                  batch_size=20,
                  epochs=200,
                  verbose=1)
model.save(PATH_TO_FILE+"convo.h5")



model = Sequential()
model.add(Flatten())
model.add(Dense(128,
                 activation='relu'))
model.add(Dense(256,
                 activation='relu'))
model.add(Dense(69,
                 activation='relu'))
model.add(Dense(14, 
                activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

model.fit(xs, ys,
          batch_size=30,
          epochs=800,
          verbose=1)

model.save(PATH_TO_FILE+"Dense.h5")
