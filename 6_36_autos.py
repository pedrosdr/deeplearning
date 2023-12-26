import pandas as pd
import numpy as np
import sklearn as skl
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from keras.layers import Dense, BatchNormalization, Dropout
from keras.initializers import HeNormal
from keras.models import Sequential
from keras.optimizers import Adam
from keras.optimizers.schedules import ExponentialDecay
import keras
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import GridSearchCV
from keras.regularizers import L2
from sklearn.preprocessing import StandardScaler

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
df: pd.DataFrame = pd.read_csv('autos.csv', encoding='latin1')

# Entendendo os parametros
df['name'].value_counts()
df['seller'].value_counts()
df['offerType'].value_counts()

# Excluindo colunas não necessárias
df = df.drop(columns=[
    'dateCrawled', 
    'dateCreated', 
    'postalCode', 
    'nrOfPictures', 
    'lastSeen', 
    'name', 
    'seller', 
    'offerType'])

# Removendo outliers
q1 = df['price'].quantile(0.25)
q3 = df['price'].quantile(0.75)
iqr = q3 - q1

mask_outliers = (df['price'] < q1 - 0.1 * iqr) | (df['price'] > q3 + 1.5 * iqr)
outliers = df[mask_outliers]
df = df.drop(outliers.index)

# Tratando valores nulos
# df['vehicleType'].value_counts().idxmax()

df = df.fillna({
    'vehicleType': df['vehicleType'].mode()[0],
    'gearbox': df['gearbox'].mode()[0],
    'notRepairedDamage': df['notRepairedDamage'].mode()[0]
})

# Separando x e y
x = df.iloc[:,1:].to_numpy()[:10000,:]
y = df.iloc[:,0].to_numpy()[:10000]

# LabelEncoder
encoder = BatchLabelEncoder([0, 1, 3, 5, 8, 9, 10])
encoder.fit(x)
x = encoder.transform(x)
    
# OneHotEncoder e StandardScaler
ohencoder = ColumnTransformer(
    [
         ('one_hot_encoder', OneHotEncoder(), [0, 1, 3, 5, 8, 9, 10])
    ],
    remainder='passthrough'
)
x = ohencoder.fit_transform(x).toarray()

# # StandardScaler
stdx = ColumnTransformer(
    [
         ('standard_scaler', StandardScaler(), [292, 293, 294, 295])
    ],
    remainder='passthrough'
)
x = stdx.fit_transform(x)

# stdsc = StandardScaler()
# stdsc.fit(x)
# x = stdsc.transform(x)

# Dividindo a base
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2)

# # Estrutura da rede neural
# model = Sequential()
# model.add(Dense(158, activation='elu', input_shape=(317,)))
# # model.add(BatchNormalization())
# # model.add(Dropout(0.2))
# for i in range(3):
#     model.add(Dense(158, activation='elu'))
#     # model.add(BatchNormalization())
#     # model.add(Dropout(0.2))
# model.add(Dense(1, activation='linear'))

# lr = ExponentialDecay(0.001, 80000, 0.96)
# model.compile(optimizer=Adam(0.001), loss='mean_absolute_error', metrics=['mean_absolute_error'])

# # Treinando a rede neural
# model.fit(xtrain, ytrain, batch_size=800, epochs=300)

# # Fazendo as previsões
# y_new = model.predict(xtest)

# print(mean_absolute_error(ytest, y_new))


# Função getModel
def getModel(loss = 'mean_absolute_percentage_error'):
    model = Sequential()
    model.add(Dense(140, activation='relu', input_shape=(x.shape[1],)))
    model.add(Dropout(0.2))
        
    for i in range(1):
        model.add(Dense(158, activation='relu'))
        model.add(Dropout(0.2))
        
    model.add(Dense(1, activation='linear'))
    
    model.compile(optimizer='adam', loss=loss, metrics=['mean_absolute_error'])
    return model


# reg = getModel('mean_absolute_error')
# reg.fit(xtrain, ytrain, batch_size=800, epochs=1000)

# # Fazendo as previsões
# y_new = reg.predict(xtest)

# print(mean_absolute_error(ytest, y_new))

reg = KerasRegressor(model = getModel, batch_size=800, epochs=1000)

params = {
    'model__loss': 
        [
            'mean_squared_error' , 
            'mean_absolute_error' , 
            'mean_absolute_percentage_error' , 
            'mean_squared_logarithmic_erro',
            'squared_hinge'
        ]
}
    
search = GridSearchCV(reg, params, scoring='neg_mean_absolute_error', cv=3)
search.fit(x, y)

search.best_params_
search.best_score_

# Testing best params
reg = getModel('squared_hinge')
reg.fit(xtrain, ytrain, batch_size=800, epochs=3000)

# Fazendo as previsões
y_new = reg.predict(xtest)

print(mean_absolute_error(ytest, y_new))
print(r2_score(ytest, y_new))
