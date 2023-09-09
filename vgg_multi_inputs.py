from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16
from keras.layers import ZeroPadding2D, Convolution2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.models import Sequential
import os
import cv2
import numpy as np
import random

model = VGG16()
model.compile(optimizer='adam', loss='categorical_crossentropy')

categories = ['dog_test', 'cat_test']
testingDIR = './new_test'
IMG_size=224
testing_data = []
for category in categories:
    path = os.path.join(testingDIR, category)
    label = categories.index(category)

    for img in os.listdir(path):
        try:
            img=path+'/'+img
            img_array = cv2.imread(img, cv2.IMREAD_COLOR)
            new_array = cv2.resize(img_array, (IMG_size, IMG_size))
            testing_data.append([new_array, label])
        except Exception as e:
            pass

random.shuffle(testing_data)

features_test = []
labels_test = []

for features, labels in testing_data:
    features_test.append(features)
    labels_test.append(labels)

features_test = np.array(features_test).reshape(-1, IMG_size, IMG_size, 3)
features_test = features_test/255.0

y_pred = model.predict(features_test)
scores = model.evaluate(features_test, labels_test)
print('%s (%.2f%%)' % (scores[0], scores[1]*100))






















