# This Python file uses the following encoding: utf-8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import DBSCAN

from sklearn.datasets.samples_generator import make_blobs
from sklearn.neighbors import NearestNeighbors

dataset = pd.read_csv("dataset.csv")

# Convertendo string em float através de representação
#   por exemplo, um grupo de item=pizza representa um número, assim, pode ser contabilizado pelo algoritmo, em vez de trabalhar com string tabalha com número
labelEncoder = LabelEncoder()
labelEncoder.fit(dataset['id_transacao'])
dataset['id_transacao'] = labelEncoder.transform(dataset['id_transacao'])

# convert the 'Date' column to datetime format
dataset['horario_pedido']= pd.to_datetime(dataset['horario_pedido'])

dataset['nova_data'] = [d.date() for d in dataset['horario_pedido']]
dataset['nova_hora'] = [d.time() for d in dataset['horario_pedido']]

# convert the 'Date' column to datetime format
dataset['nova_data']= pd.to_datetime(dataset['nova_data'])

# Divisão dos dados em variáveis dependentes (X) e independentes (Y)
# Dependentes (X): id_transacao,quantidade_item
# Independentes (y): horario_pedido,localidade,nome_item,latitude,longitude,nova_data,nova_hora
idx = pd.IndexSlice
#df.loc[idx[:, :, 'C1', :],:]
XX = dataset.loc[:,idx['id_transacao','quantidade_item']]
yy = dataset.loc[:,idx['localidade','nome_item','latitude','longitude','nova_data','nova_hora']]

#print(XX.dtypes)

X, y = make_blobs(n_samples=300, centers=XX, cluster_std=0.60, random_state=0)
#plt.scatter(X[:,0], X[:,1])

neigh = NearestNeighbors(n_neighbors=2)
nbrs = neigh.fit(X)
distances, indices = nbrs.kneighbors(X)
#print(indices)

distances = np.sort(distances, axis=0)
distances = distances[:,1]
#plt.plot(distances)

#plt.show()

m = DBSCAN(eps=4, min_samples=5)
m.fit(X)

clusters = m.labels_

colors = ['royalblue', 'maroon', 'forestgreen', 'mediumorchid', 'tan', 'deeppink', 'olive', 'goldenrod', 'lightcyan', 'navy']
vectorizer = np.vectorize(lambda x: colors[x % len(colors)])

plt.scatter(X[:,0], X[:,1], c=vectorizer(clusters))

plt.show()