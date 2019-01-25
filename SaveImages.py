# -*- coding: utf-8 -*-
"""
@author: Paweł Ciepliński
"""
import time

PATH_TO_FILES = 'D:/Studia/RealTimeSignRecognition/signs_da/'

def array_to_coords(array, label):
    coords = label + '\nsnap_1'
    for i in range(0,127):
        for j in range(0,128):
            if (array[i][j] != 0):
                coords = coords + '\n' + str(i) + ' ' + str(j)
    return coords
            
def getLabel(numb):
    if(numb == 0):
        return "accident"
    elif numb == 1 :
        return "bomb"
    elif numb == 2 :
        return "car"
    elif numb == 3 :
        return "casualty"
    elif numb == 4 :
        return "electricity"
    elif numb == 5 :
        return "fire"
    elif numb == 6 :
        return "fire_brigade"
    elif numb == 7 :
        return "flood"
    elif numb == 8 :
        return "gas"
    elif numb == 9 :
        return "injury"
    elif numb == 10 :
        return "paramedics"
    elif numb == 11 :
        return "person"
    elif numb == 12 :
        return "police"
    elif numb == 13 :
        return "roadblock"

def save_to_file(x, y):
    label = getLabel(y)
    array = x.reshape(128, 128)

    coords = array_to_coords(array, label)
    
    path = PATH_TO_FILES + label + "/" + label + '_' + str(int(time.time()*1000.0)) + '.txt'
    
    file = open(path, "w+")
    file.write(coords)
    file.close()