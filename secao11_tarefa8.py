from keras.layers import Dense, Flatten
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.metrics import accuracy_score

# Carregando a base
df = pd.read_csv('personagens.csv')

# Separando as variáveis
x = df.iloc[:,:6]
y = df.iloc[:,6]

# Label Encoding y
le = LabelEncoder()
le.fit(y)
y = le.transform(y) # Bart:0   Homer: 1

# Padronizando x
stdx = StandardScaler()
stdx.fit(x)
x = stdx.transform(x)

# Separando as bases
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.25)

# Estrutura da rede
model = Sequential()
model.add(Dense(3, activation='elu', input_shape=(6,)))
model.add(Dense(3, activation='elu'))
model.add(Dense(3, activation='elu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Treinando a red
model.fit(xtrain, ytrain, batch_size=100, epochs=1000, validation_data=[xtest, ytest])

# Fazendo as previsões
y_new = np.array([0 if x < 0.5 else 1 for x in model.predict(xtest)])
print('Acurácia do modelo: ', accuracy_score(ytest, y_new))
