# This Python file uses the following encoding: utf-8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import DBSCAN

dataset = pd.read_csv("dataset.csv")

# Convertendo string em float através de representação
#   por exemplo, um grupo de item=pizza representa um número, assim, pode ser contabilizado pelo algoritmo, em vez de trabalhar com string tabalha com número
labelEncoder = LabelEncoder()
labelEncoder.fit(dataset['nome_item'])
dataset['nome_item'] = labelEncoder.transform(dataset['nome_item'])

labelEncoder.fit(dataset['horario_pedido'])
dataset['horario_pedido'] = labelEncoder.transform(dataset['horario_pedido'])

labelEncoder.fit(dataset['id_transacao'])
dataset['id_transacao'] = labelEncoder.transform(dataset['id_transacao'])

# Data preparation
np_dataset3 = dataset[].
np_dataset3 = np_dataset3.astype(np.double)
CustoAquisicao = np_dataset3[:,0]
PrecoVenda = np_dataset3[:,1]
CustoTotalVendas = np_dataset3[:,2]
# Run DBSCAN algorithm
model = DBSCAN (eps=0.3, min_samples=3, algorithm='auto').fit(np_dataset3)
model.labels_ #displays the cluster number for each data entry
# Data Vizualization
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(CustoAquisicao, PrecoVenda, CustoTotalVendas, c=model.labels_,
marker='o')
ax.set_xlabel('Custo de Aquisição')
ax.set_ylabel('Preço de Venda')
ax.set_zlabel('Custo Total de Venda')
plt.show()