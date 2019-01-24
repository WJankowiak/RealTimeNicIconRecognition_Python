# -*- coding: utf-8 -*-
"""
@author: Paweł Ciepliński
"""
import keras
import numpy as np
from keras.utils import to_categorical
import ReadImages as RI

IMAGES_DIRECTORY = 'D:/Studia/RealTimeSignRecognition/signs_test'
np.set_printoptions(threshold=np.nan)

ys, xs = RI.load_data_from_coords(IMAGES_DIRECTORY)   
xs = np.reshape(xs, (305, 16384))
xs = xs.reshape(xs.shape[0], 128, 128, 1)
xs = xs.astype('int32')
ys = to_categorical(ys, 14)
model = keras.models.load_model("D:/Studia/RealTimeSignRecognition/convo.h5")

confusion_matrix = np.zeros((14,14))

for i in range (0,305):
    x = xs[i]
    x = x.reshape(1,128,128,1)
    res = model.predict(x)
    predicted = res[0].argmax(axis=0)
    expected = ys[i].argmax(axis=0)
    confusion_matrix[expected][predicted] += 1 
print (confusion_matrix)

