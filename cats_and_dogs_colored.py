import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.layers import Conv2D

### Training and testing values
DATADIR = "D:\python-projects\\Neural_networks\\train"
Categories = ['dogs', 'cats']

X_train = []
y_train = []
X_test = []
IMG_size = 100

### Get and sort training data
training_data = []
def create_training_data():
    for category in Categories:  # do dogs and cats

        path = os.path.join(DATADIR,category)  # create path to dogs and cats
        class_num = Categories.index(category)  # get the classification  (0 or a 1). 0=dog 1=cat

        for img in tqdm(os.listdir(path)):  # iterate over each image per dogs and cats
            try:
                img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_COLOR)  # convert to array
                new_array = cv2.resize(img_array, (IMG_size, IMG_size))  # resize to normalize data size
                training_data.append([new_array, class_num])  # add this to our training_data
            except Exception as e:  # in the interest in keeping the output clean...
                pass
create_training_data()

random.shuffle(training_data)

### retrieve testing data
testing_data = []

testingDIR = "D:\python-projects\\Neural_networks\\test"

for img in tqdm(os.listdir(testingDIR)):
    testing_array = cv2.imread(os.path.join(testingDIR, img), cv2.IMREAD_COLOR)
    new_testing_array = cv2.resize(testing_array, (IMG_size, IMG_size))
    testing_data.append(new_testing_array)

### distribute training data and testing data

for features, labels in training_data:
    X_train.append(features)
    y_train.append(labels)

for features in testing_data:
    X_test.append(features)

### resize training and testing features

X_train = np.array(X_train).reshape(-1, IMG_size, IMG_size, 3)
X_train = X_train/255.0

X_test = np.array(X_test).reshape(-1, IMG_size, IMG_size, 3)
X_test = X_test/255.0

y_train = np.array(y_train)



### creating the neural network

model = Sequential()
model.add(BatchNormalization(axis=1))
model.add(Conv2D(8, (3,3), input_shape=X_train.shape[1:]))
model.add(BatchNormalization(axis=1))
model.add(Conv2D(16, (3,3), activation='relu', padding='same', strides=2))
model.add(Dropout(0.25))
model.add(BatchNormalization(axis=1))
model.add(Conv2D(16, (3,3), activation='relu'))
model.add(BatchNormalization(axis=1))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(Dropout(0.25))
model.add(BatchNormalization(axis=1))
model.add(Conv2D(64, (3,3), activation='relu', strides=2, padding='same'))
model.add(BatchNormalization(axis=1))
model.add(Conv2D(128, (3,3), activation='relu'))
model.add(BatchNormalization(axis=1))
model.add(Conv2D(128, (3,3), activation='relu', strides=2, padding='same'))
model.add(BatchNormalization(axis=1))
model.add(Dropout(0.25))
model.add(Conv2D(256, (3,3), activation='relu'))
model.add(BatchNormalization(axis=1))
model.add(Conv2D(256, (3,3), activation='relu'))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, batch_size=64, epochs=12, verbose=1, validation_split=0.2, shuffle=True)

model.summary()
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


### test model
y_pred = model.predict(X_test, batch_size=64, verbose=2)

print(y_pred[:10])

model.save("cats_and_dogs_colored_CNN.h5")
print("Saved model to D:\python-projects\projects")














