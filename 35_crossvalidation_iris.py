import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from keras.utils import to_categorical
from keras.initializers import HeNormal
from keras.layers import BatchNormalization, Dropout
import keras
from keras.regularizers import L1
from sklearn.metrics import accuracy_score, confusion_matrix
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import cross_val_score

df = pd.read_csv('iris.csv')
x = df.iloc[:,:4].to_numpy()
y = df.iloc[:,4].to_numpy()

le = LabelEncoder()
le.fit(y)
y = le.transform(y)

y_dummy = to_categorical(y)

def getModel():
    rn = Sequential()
    
    rn.add(Dense(3, 
                 activation='elu', 
                 kernel_initializer=HeNormal(), 
                 kernel_regularizer=L1(),
                 input_shape = (4,)))
    rn.add(BatchNormalization())
    for i in range(2):
        rn.add(Dense(3, 
                     activation='elu', 
                     kernel_initializer=HeNormal(), 
                     kernel_regularizer=L1()))
        rn.add(BatchNormalization())
    
        
    rn.add(Dense(3, activation='softmax', kernel_initializer=HeNormal()))
    lr = keras.optimizers.schedules.ExponentialDecay(0.001, 10000, 0.0001)
    optm = keras.optimizers.Adam(lr)
    rn.compile(optimizer=optm, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    
    return rn

clf = KerasClassifier(
    model=getModel,
    batch_size=300,
    epochs=3000
)

results = cross_val_score(clf, x, y_dummy, scoring='accuracy', cv=5)

print(results.mean())
print(results.std())

