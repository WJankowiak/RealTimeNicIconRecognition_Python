# -*- coding: utf-8 -*-
"""
@author: Paweł Ciepliński
"""
import numpy as np
import ReadImages as RI
import SaveImages as SI

IMAGES_DIRECTORY = 'D:/Studia/RealTimeSignRecognition/signs'
DATASET_SIZE = 1972

np.set_printoptions(threshold=np.nan)

ys, xs = RI.load_data_from_coords(IMAGES_DIRECTORY)  

xs = np.reshape(xs, (DATASET_SIZE, 16384))
xs = xs.reshape(xs.shape[0], 1, 128, 128)

from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        fill_mode='nearest')

datagen.fit(xs)

for xs_batch, ys_batch in datagen.flow(xs, ys, batch_size=DATASET_SIZE):
	for i in range(0, DATASET_SIZE):
		SI.save_to_file(xs_batch[i], ys_batch[i])
	break