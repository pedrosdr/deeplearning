from keras.datasets import cifar10
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, BatchNormalization, Dropout
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score, confusion_matrix

(xtrain, ytrain), (xtest, ytest) = cifar10.load_data()

xtrain = xtrain.astype(np.float32)
ytrain = to_categorical(ytrain.astype(np.float32))
xtest = xtest.astype(np.float32)
ytest = to_categorical(ytest.astype(np.float32))

# Plotando a imagem
xt = xtrain
# xt = xtrain.mean(axis=3, keepdims=False)
xt = 0.299 * xt[:,:,:,0] + 0.587 * xt[:,:,:,1] + 0.114 * xt[:,:,:,2]
plt.imshow(xt[1011], cmap='gray')

# Estrutura da rede neural
model = Sequential()
model.add(Conv2D(32, (3,3), input_shape=(32,32,3), activation='relu'))
model.add(BatchNormalization())

model.add(MaxPool2D())

model.add(Conv2D(32, (3,3), activation='relu'))
model.add(BatchNormalization())

model.add(MaxPool2D())
model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(90, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(90, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(10, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics='accuracy')

# Treinando o modelo
model.fit(xtrain, ytrain, batch_size=128, epochs=50, validation_data=(xtest, ytest))

# Fazendo as previsões
y_new = model.predict(xtest)

y_new2 = [x.argmax() for x in y_new]
ytest2 = [x.argmax() for x in ytest]

# Testando as previsões
print("Accuracy: ", accuracy_score(ytest2, y_new2))
plt.imshow(confusion_matrix(ytest2, y_new2))
