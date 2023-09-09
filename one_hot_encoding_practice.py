import numpy as np
from sklearn import preprocessing
data = ['Cat', 'Dog', 'Cat', 'Rabbit', 'Chicken', 'Dog', 'Cat', 'Elephant']
data = np.array(data)
print(data)

le = preprocessing.LabelEncoder()
integer_encoded = le.fit_transform(data)
print(integer_encoded)

onehot = preprocessing.OneHotEncoder(sparse=False, categories='auto')
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehotencoded = onehot.fit_transform(integer_encoded)
print(onehotencoded)







