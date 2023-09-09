from keras.layers import ZeroPadding2D, Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from keras.models import Sequential
import os
from keras.optimizers import sgd
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt

print(1)
DATADIR = "./train"
Categories = ['dogs', 'cats']
X_train = []
y_train = []
IMG_size = 100
training_data = []
training_data = []
for category in Categories:  # do dogs and cats

    path = os.path.join(DATADIR,category)  # create path to dogs and cats
    class_num = Categories.index(category)  # get the classification  (0 or a 1). 0=dog 1=cat

    for img in os.listdir(path):  # iterate over each image per dogs and cats
        try:
            img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_COLOR)  # convert to array
            new_array = cv2.resize(img_array, (IMG_size, IMG_size))  # resize to normalize data size
            training_data.append([new_array, class_num])  # add this to our training_data
        except Exception as e:  # in the interest in keeping the output clean...
            pass

print(1)
random.shuffle(training_data)
for features, labels in training_data:
    X_train.append(features)
    y_train.append(labels)

X_train = np.array(X_train).reshape(-1, IMG_size, IMG_size, 3)
X_train = X_train / 255.0

y_train = np.array(y_train)
print('dataset finished')

model = Sequential()

model.add(ZeroPadding2D((1, 1), input_shape=(100, 100, 3)))
model.add(BatchNormalization(axis=1))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization(axis=1))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2), strides=2))

model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2), strides=2))

model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2), strides=2))

model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Conv2D(512, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2), strides=2))


model.add(Flatten()) #7x7x512->2048
model.add(Dense(2048, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(256, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='softmax'))
model.summary()
model.compile(optimizer=sgd(lr=0.2), loss='binary_crossentropy', metrics=['accuracy'])
print(1)
history = model.fit(X_train, y_train, batch_size=64, epochs=17, verbose=1, validation_split=0.2, shuffle=True)

categories = ['dog_test', 'cat_test']
testingDIR = './new_test'
IMG_size = 100
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



print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


















