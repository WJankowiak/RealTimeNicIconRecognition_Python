# -*- coding: utf-8 -*-
"""
@author: Wojciech Jankowiak
"""
from os import path, walk
from copy import deepcopy
import numpy as np

def load_data_from_coords(IMAGES_DIRECTORY):
    data = []
    labels = []
    for current_dir, dirs, filenames in walk(IMAGES_DIRECTORY):
      label = path.basename(path.normpath(current_dir))
      converted = convertLabel(label)
      for filename in filenames:
          extracted = extractData(path.join(current_dir, filename))
          for part in extracted:
              image = load_image_from_coord(part)
              data.append(image)
              labels.append(converted)
    return labels, np.asarray(data)

def load_image_from_coord(part):
    image_template = np.zeros((128,128))
    for x_coord, y_coord in part:
        if x_coord <128 and y_coord <128: 
            image_template[x_coord, y_coord] = 1
    return image_template

def extractData(file):
    extracted =[]
    with open(file) as f:
        f.readline()
        f.readline()
        part = []
        for line in f:
          if 'snap' in line :
              c = deepcopy(part)
              extracted.append(c)
          else:
            part.append([int(x) for x in line.split()])
    return extracted

def convertLabel(label):
    if(label == "accident"):
        return 0
    elif label == "bomb" :
        return 1
    elif label == "car" :
        return 2
    elif label == "casualty" :
        return 3
    elif label == "electricity" :
        return 4
    elif label == "fire" :
        return 5
    elif label == "fire_brigade" :
        return 6
    elif label == "flood" :
        return 7
    elif label == "gas" :
        return 8
    elif label == "injury" :
        return 9
    elif label == "paramedics" :
        return 10
    elif label == "person" :
        return 11
    elif label == "police" :
        return 12
    elif label == "roadblock" :
        return 13