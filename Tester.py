# -*- coding: utf-8 -*-
"""
@author: Wojciech Jankowiak
"""
import keras
import numpy as np
from keras.utils import to_categorical
import ReadImages as RI

IMAGES_DIRECTORY = 'E:/Studia/RealTimeSignRecognition/signs_test'
np.set_printoptions(threshold=np.nan)

ys, xs = RI.load_data_from_coords(IMAGES_DIRECTORY)   
xs = np.reshape(xs, (305, 16384))
xs = xs.reshape(xs.shape[0], 128, 128, 1)
xs = xs.astype('int32')
ys = to_categorical(ys, 14)
model = keras.models.load_model("E:/STUDIA/Convo.h5")
testres=0
for i in range (0,305):
    x = xs[i]
    x = x.reshape(1,128,128,1)
    res = model.predict(x)
    if(np.argmax(res) == np.argmax(ys[i])):
        testres = testres+1
print (testres/305)
