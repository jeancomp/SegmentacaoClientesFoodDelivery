# This Python file uses the following encoding: utf-8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder

dataset = pd.read_csv("dataset.csv")

# Plotando os dados latitude e longitude
plt.scatter(dataset.iloc[:,5], dataset.iloc[:,6]) #posicionamento dos eixos x e y
plt.xlim(4, 42) #range do eixo x
plt.ylim(-80, -89) #range do eixo y
plt.grid() #função que desenha a grade no nosso gráfico
plt.show()

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
# Dependentes (X): id_transacao,quantidade_item
# Independentes (y): horario_pedido,localidade,nome_item,latitude,longitude,nova_data,nova_hora
idx = pd.IndexSlice
#df.loc[idx[:, :, 'C1', :],:]
X = dataset.loc[:,idx['id_transacao','quantidade_item']]
y = dataset.loc[:,idx['localidade','nome_item','latitude','longitude','nova_data','nova_hora']]

print("\nVariáveis Dependentes:")
print(X.head(5))
print("\nVariáveis Independentes:")
print(y.head(5))

# Representar os id_transacao por número
labelEncoder = LabelEncoder()
labelEncoder.fit(X['id_transacao'])
X['id_transacao'] = labelEncoder.transform(X['id_transacao'])

# # Método Elbow
# wcss = []
#
# print("\n Encontrar valor ideal de k(nº de clustering ideal): ")
# for i in range(1, 11):
#     kmeans = KMeans(n_clusters=i, init='random')
#     kmeans.fit(X)
#     print i, kmeans.inertia_
#     wcss.append(kmeans.inertia_)
# plt.plot(range(1, 11), wcss)
# plt.title('O Metodo Elbow')
# plt.xlabel('Numero de Clusters')
# plt.ylabel('WSS')
# plt.show()

kmeans = KMeans(n_clusters=4, random_state=0).fit(X)
X_clustered = kmeans.fit_predict(X)

results = X[['id_transacao']].copy()
results['clusterNumber'] = X_clustered
print("\nCluster encontrados:")
print(results)

# Gráfico de barras
# plt.bar(results['clusterNumber'], results['clusterNumber'].count())
# plt.title('O Metodo Elbow')
# plt.xlabel('clusterNumber')
# plt.ylabel('Total cluster')
# plt.show()

# LABEL_COLOR_MAP = {0 : 'red', 1 : 'blue', 2: 'green', 3: 'orange', 4: 'purple'}
# label_color = [LABEL_COLOR_MAP[l] for l in X_clustered]
#
# c1 = 0 # valor do índice da coluna, pode ser 0, 1 ou 2
# c2 = 1
# labels = ['sepal length', 'sepal width', 'petal length']
# c1label = labels[c1]
# c2label = labels[c2]
# title = c1label + ' x ' + c2label
#
# plt.figure(figsize = (10,10))
# plt.scatter(results.iloc[:, c1], results.iloc[:, c2], c=label_color, alpha=0.3)
# plt.xlabel(c1label, fontsize=18)
# plt.ylabel(c2label, fontsize=18)
# plt.suptitle(title, fontsize=20)
# #plt.savefig(title + '.jpg')
# plt.show()