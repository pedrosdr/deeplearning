import pandas as pd
from minisom import MiniSom
from sklearn.preprocessing import MinMaxScaler
from pylab import pcolor, colorbar, plot

df = pd.read_csv('wines.csv')
x = df.iloc[:,1:14].to_numpy()
y = df.iloc[:,0].to_numpy()

scaler = MinMaxScaler()
x = scaler.fit_transform(x)

som = MiniSom(x=8, y=8, input_len=13)
som.random_weights_init(x)
som.train_random(data=x, num_iteration=100)

som._weights
q = som.activation_response(x)

# Results
markers = ['o', 's', 'D']
colors = ['r', 'g', 'b']

y -= 1

# Mean inter-neuron distance
pcolor(som.distance_map())
colorbar()

for i, xi in enumerate(x):
    w = som.winner(xi)
    plot(w[0] + 0.5, w[1] + 0.5, markers[y[i]],
         markerfacecolor = 'None',
         markeredgecolor = colors[y[i]]
    )