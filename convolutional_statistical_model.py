import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Input, Reshape, Flatten
from keras.layers import Dense, Dropout
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_score

(xtrain, ytrain), (xtest, ytest) = keras.datasets.cifar10.load_data()

xtrain = xtrain[np.isin(ytrain.flatten(), [7,9])]
ytrain = ytrain[np.isin(ytrain.flatten(), [7,9])]

xtest = xtest[np.isin(ytest.flatten(), [7,9])]
ytest = ytest[np.isin(ytest.flatten(), [7,9])]

ytrain = np.where(ytrain == 7, 1, 0)
ytest = np.where(ytest == 7, 1, 0)


convolutional_part = Sequential([
    Input([32,32,3]),
    
    Conv2D(30, [3,3], [1,1], 'same', activation='relu'),
    BatchNormalization(),
    
    Conv2D(30, [3,3], [1,1], 'same', activation='relu'),
    BatchNormalization(),
    
    MaxPooling2D(),
    
    Conv2D(60, [3,3], [1,1], 'same', activation='relu'),
    BatchNormalization(),
    
    MaxPooling2D(),
    
    Conv2D(100, [3,3], [1,1], 'same', activation='relu'),
    BatchNormalization(),
    
    MaxPooling2D()
])

dense_part = Sequential([
    Input([4,4,100]),
    
    Flatten(),
    Dense(800, activation='relu'),
    Dropout(0.4),
    
    Dense(100, activation='relu'),
    Dropout(0.4),
    
    Dense(1, activation='sigmoid')
])

full_model = Sequential([
    convolutional_part,
    dense_part
])

full_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])

full_model.fit(xtrain, ytrain, batch_size=100, epochs=100)

full_model.save('convolutional_statistical_model.keras')

# Testing model
res = full_model.predict(xtest)
ypred = np.array([1 if x > 0.5 else 0 for x in res]).reshape(-1,1)

confusion_matrix(ytest, ypred)
accuracy_score(ytest, ypred)
recall_score(ytest, ypred)
precision_score(ytest, ypred)


# Statistical Model
vxtrain = convolutional_part.predict(xtrain)
vxtrain = vxtrain.reshape(vxtrain.shape[0], 4*4*100)

vxtest = convolutional_part.predict(xtest)
vxtest = vxtest.reshape(vxtest.shape[0], 4*4*100)

lr = LogisticRegression()
lr.fit(vxtrain, ytrain.flatten())

ypred = lr.predict(vxtest)
confusion_matrix(ytest, ypred)
accuracy_score(ytest, ypred)
recall_score(ytest, ypred)
precision_score(ytest, ypred)
