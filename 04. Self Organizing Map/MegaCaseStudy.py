#%% Mega Case study - Make a Hybrid Deep Learning Model

#%%% Part 1 - Identify the Frauds with the self-Organizing Map

#%%%% Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%%%% Importing the dataset
dataset = pd.read_csv('Credit_Card_Applications.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#%%%% Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0, 1))
X = sc.fit_transform(X)

#%%%% Training the SOM
from minisom import MiniSom
som = MiniSom(x=10, y=10, input_len=15, sigma=1.0, learning_rate=0.5)
som.random_weights_init(X)
som.train_random(data=X, num_iteration=100)

#%%%% Visualizing the results
from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som.distance_map().T)
colorbar()
markers = ['o', 's']
colors = ['r', 'g']
for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0] + 0.5,
         w[1] + 0.5,
         markers[y[i]],
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2)
show()

#%%%% Finding the frauds
mappings = som.win_map(X)
frauds = np.concatenate((mappings[(4, 2)], mappings[(5,1)]), axis=0)
frauds = sc.inverse_transform(frauds)

#%%% Part 2 - Going from unsupervised to Supervised Deep Learing

#%%%% Creating the matrix of feature
customers = dataset.iloc[:, 1:].values

#%%%% Creating the dependent variable
is_fraud = np.zeros(len(dataset))
for i in range(len(dataset)):
    if dataset.iloc[i, 0] in frauds:
        is_fraud[i] = 1

#%%%% Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
customers = sc.fit_transform(customers)

#%%%% Importing the Keras libraries and package
from keras.models import Sequential
from keras.layers import Dense

#%%%% Initialising the ANN
ann = Sequential()

#%%%% Adding the input and the first hidden layer
ann.add(Dense(units=2, activation='relu', input_dim=15))
    
#%%%% Adding the output layer
ann.add(Dense(units=1, activation='sigmoid'))

#%%%% Compiling the ANN
ann.compile(optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy'])

#%%%% Fitting the ANN to the Training set
ann.fit(customers, is_fraud, batch_size=1, epochs=2)

#%%%% Predicting the probabilities of frauds
y_pred = ann.predict(customers)
y_pred = np.concatenate((dataset.iloc[:, 0:1].values, y_pred), axis=1)
y_pred = y_pred[y_pred[:, 1].argsort()]