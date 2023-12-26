import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

# Criando classe para encoding
class BatchLabelEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, indexes):
        self.encoders = []
        self.indexes = indexes;
        
    def fit(self, X, y=None):
        for i in self.indexes:
            self.encoders.append(LabelEncoder().fit(X[:,i]))
        return self
    
    def transform(self, X, y=None):
        for i in range(len(self.indexes)):
            X[:,self.indexes[i]] = self.encoders[i].transform(X[:,self.indexes[i]])
        return X

# Carregando a base
df: pd.DataFrame = pd.read_csv('games.csv')

# Excluindo colunas
df = df.drop(columns=['Other_Sales', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Developer', 'Name', 'Publisher'])



# Tratando dados faltantes (categóricos)
df = df.dropna()

# Criando x e y
y = df.iloc[:,3].to_numpy()
x = df.iloc[:,[0,1,2,4,5,6,7,8]].to_numpy()

# Fazendo label encoding das variaveis categóricas
le = BatchLabelEncoder([0,2,7])
le.fit(x)    
x = le.transform(x)

# Removendo 'tbd' do array
x = np.where(x == 'tbd', np.nan, x).astype(np.float64)

# Aplicando OneHotEncoder
onehot = ColumnTransformer(
    [
     ('onehotencoder', OneHotEncoder(), [0,2,7])
    ],
    remainder='passthrough'
)
onehot.fit(x)
x = onehot.transform(x).toarray()

# Escalonamento
stdsc = ColumnTransformer(
    [
      ('standardscaler', StandardScaler(), [36,37,38,39,40])
    ],
    remainder='passthrough'
)
stdsc.fit(x)
x = stdsc.transform(x)

###############################################################################
#                            Criando o modelo

from keras.models import Model
from keras.layers import Dense, BatchNormalization, Dropout, Input, Activation
from sklearn.metrics import r2_score
from keras import backend as K
from sklearn.model_selection import cross_val_score, train_test_split
from scikeras.wrappers import KerasRegressor

# Criando a função customizada (leaking relu)
def custom_function(x):
    return K.switch(x < 0, x * 0.05, x)

# Montando a rede neural
def getModel():
    input_layer = Input((x.shape[1]))
    hidden1 = Dense(20, activation=Activation(custom_function))(input_layer)
    bn1 = BatchNormalization()(hidden1)
    dropout1 = Dropout(0.1)(bn1)
    hidden2 = Dense(10, activation=Activation(custom_function))(dropout1)
    bn2 = BatchNormalization()(hidden2)
    dropout2 = Dropout(0.1)(bn2)
    hidden3 = Dense(5, activation=Activation(custom_function))(dropout2)
    bn3 = BatchNormalization()(hidden3)
    dropout3 = Dropout(0.1)(bn3)
    output_layer = Dense(1, activation='linear')(dropout3)

    model = Model(inputs=input_layer, outputs=output_layer)

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])
    
    return model

# Criando um regressor sklearn
reg = KerasRegressor(model=getModel, batch_size=6000, epochs=5000)

# Fazendo a cross validation
results = cross_val_score(reg, x, y, cv=5, scoring='neg_mean_absolute_error')

# treinando uma única vez
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2)

model = getModel()
model.fit(xtrain, ytrain, batch_size=6000, epochs=5000)
y_new = model.predict(xtest)
print(r2_score(ytest, y_new))
