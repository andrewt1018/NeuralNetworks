import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

DATADIR = "D:\python-projects\projects\\train"

CATEGORIES = ["dogs", "cats"]

IMG_SIZE = 200

training_data = []


for category in CATEGORIES:  # do dogs and cats

    path = os.path.join(DATADIR,category)  # create path to dogs and cats
    #class_num = CATEGORIES.index(category)  # get the classification  (0 or a 1). 0=dog 1=cat

    for img in tqdm(os.listdir(path)):  # iterate over each image per dogs and cats

            img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_COLOR)  # convert to array
            print(img_array)
            new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
            print("###############################################","\n")
            print(new_array)
            plt.imshow(new_array)
            plt.show()
            continue
            break
    break


