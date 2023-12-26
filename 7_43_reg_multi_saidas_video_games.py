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
df = df.drop(columns=['Other_Sales', 'Global_Sales', 'Developer', 'Name', 'Publisher'])



# Tratando dados faltantes (categóricos)
df = df.fillna({
    'Platform': df['Platform'].mode()[0],
    'Year_of_Release': df['Year_of_Release'].mode()[0],
    'Genre': df['Genre'].mode()[0],
    'Rating': df['Rating'].mode()[0]
})

# Criando x e y
y = df.iloc[:,3:6].to_numpy()
x = df.iloc[:,[0,1,2,6,7,8,9,10]].to_numpy()

# Fazendo label encoding das variaveis categóricas
le = BatchLabelEncoder([0,2,7])
le.fit(x)    
x = le.transform(x)

# Removendo 'tbd' do array
x = np.where(x == 'tbd', np.nan, x).astype(np.float64)

# Tratando dados faltantes variáveis não categóricas
imputer = SimpleImputer()
x = imputer.fit_transform(x)
y = imputer.fit_transform(y)

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
     ('standardscaler', StandardScaler(), [51,52,53,54,55])
    ],
    remainder='passthrough'
)
stdsc.fit(x)
x = stdsc.transform(x)

# Separando os y
y = (y[:,0], y[:,1], y[:,2])


###############################################################################
#                         Estrutura da Rede Neural

from keras.layers import Dense, Input, concatenate
from keras.models import Model
from sklearn.metrics import r2_score

input_layer = Input(shape=(x.shape[1],))
hidden1 = Dense(29, activation='elu')(input_layer)
hidden2 = Dense(19, activation='elu')(hidden1)

hidden3 = Dense(19, activation='elu')(input_layer)
hidden4 = Dense(29, activation='elu')(hidden3)

hidden_outputs = concatenate([hidden2, hidden4])
hidden5 = Dense(10, activation='elu')(hidden_outputs)

output_layer1 = Dense(1, activation='linear')(hidden5)
output_layer2 = Dense(1, activation='linear')(hidden5)
output_layer3 = Dense(1, activation='linear')(hidden5)

# input_layer = Input(shape=(57,))
# hidden1 = Dense(29, activation='elu')(input_layer)
# hidden2 = Dense(29, activation='elu')(hidden1)

# hidden3 = Dense(29, activation='elu')(input_layer)
# hidden4 = Dense(29, activation='elu')(hidden3)

# hidden5 = Dense(29, activation='elu')(input_layer)
# hidden6 = Dense(29, activation='elu')(hidden5)

# output_layer1 = Dense(1, activation='linear')(hidden2)
# output_layer2 = Dense(1, activation='linear')(hidden4)
# output_layer3 = Dense(1, activation='linear')(hidden6)

model = Model(inputs=input_layer, outputs=[output_layer1, output_layer2, output_layer3])
model.compile(optimizer='adam', loss='mse', metrics=['mean_absolute_error'])

model.fit(x, [y[0], y[1], y[2]], batch_size=500, epochs=3000)

y_new = model.predict(x)

print(r2_score(y[0], y_new[0]), r2_score(y[1], y_new[1]), r2_score(y[2], y_new[2]))
