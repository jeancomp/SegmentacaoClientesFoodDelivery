# This Python file uses the following encoding: utf-8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import DBSCAN

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA

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

dataset['quantidade_item'] = [d.float64() for d in dataset['quantidade_item']]

# convert the 'Date' column to datetime format
dataset['nova_data']= pd.to_datetime(dataset['nova_data'])

# Divisão dos dados em variáveis dependentes (X) e independentes (Y)
# Dependentes (X): id_transacao,quantidade_item
# Independentes (y): horario_pedido,localidade,nome_item,latitude,longitude,nova_data,nova_hora
# idx = pd.IndexSlice
# #df.loc[idx[:, :, 'C1', :],:]
# X = dataset.loc[:,idx['id_transacao','quantidade_item']]
# y = dataset.loc[:,idx['localidade','nome_item','latitude','longitude','nova_data','nova_hora']]

# Eliminando a coluna id_transacao dos dados
dataset = dataset.drop('id_transacao', axis = 1)
dataset = dataset.drop('latitude', axis = 1)
dataset = dataset.drop('longitude', axis = 1)
dataset = dataset.drop('horario_pedido', axis = 1)
dataset = dataset.drop('localidade', axis = 1)
dataset = dataset.drop('nome_item', axis = 1)
dataset = dataset.drop('nova_data', axis = 1)
dataset = dataset.drop('nova_hora', axis = 1)

# Lidando com os valores ausentes
dataset.fillna(method ='ffill', inplace = True)

print(dataset.dtypes)

# Escalonar os dados para trazer todos os atributos a um nível comparável
scaler = StandardScaler()
X_scaled = scaler.fit_transform(dataset)

#X_scaled['horario_pedido']= pd.to_datetime(dataset['horario_pedido'])

#print(X_scaled.dtypes)

#
# # Normalizando os dados para que os dados seguem aproximadamente uma distribuição Gaussiana
# X_normalized = normalize(X_scaled)
#
# # Convertendo a matriz numpy em um DataFrame do pandas
# X_normalized = pd.DataFrame(X_normalized)
#
# pca = PCA(n_components = 2)
# X_principal = pca.fit_transform(X_normalized)
# X_principal = pd.DataFrame(X_principal)
# X_principal.columns = ['P1', 'P2']
# print(X_principal.head())












