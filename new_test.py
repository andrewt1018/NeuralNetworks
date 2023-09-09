import numpy as np
import cv2
import os
from keras.models import load_model
import random
import tqdm
from keras.utils import to_categorical

model = load_model('cats_and_dogs_colored_CNN.h5')

model.summary()

categories = ['dog_test', 'cat_test']
testingDIR = './new_test'
IMG_size=100
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

y_pred = model.predict(features_test, batch_size=64, verbose=2)
scores = model.evaluate(features_test, labels_test, verbose=2)
print('%s (%.2f%%)' % (scores[0], scores[1]*100))


























