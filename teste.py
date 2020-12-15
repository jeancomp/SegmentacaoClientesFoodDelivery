# This Python file uses the following encoding: utf-8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans # 4

dataset = pd.read_csv("dataset.csv")

# convert the 'Date' column to datetime format
dataset['horario_pedido']= pd.to_datetime(dataset['horario_pedido'])

dataset['nova_data'] = [d.date() for d in dataset['horario_pedido']]
dataset['nova_hora'] = [d.time() for d in dataset['horario_pedido']]

# convert the 'Date' column to datetime format
dataset['nova_data']= pd.to_datetime(dataset['nova_data'])

print("\nAlgoritmo K-means")
print(dataset.head(5))
print(dataset.dtypes)

# Divisão dos dados em variáveis dependentes (X) e independentes (Y)
# Independentes: id_transacao,horario_pedido,localidade,nome_item,quantidade_item,latitude,longitude
# Dependentes: id_transacao,horario_pedido,quantidade_item
X = dataset.iloc[:, 0:4]
y = dataset.iloc[:, 4]

print("\nVariáveis Dependentes:")
print(X)
print("\nVariáveis Independentes:")
print(y)