# This Python file uses the following encoding: utf-8
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder

print("Algoritmos de Clustering: K-means")
print("\n")

# 3 cluster: horario pedido, localidade, nome_item
# random: Se refere ao modo de inicialização de forma aleatória,
#    ou seja, os centróides iniciais serão gerados de forma totalmente aleatória sem um critério para seleção
kmeans = KMeans(n_clusters = 5, init = 'random')

# Recebendo os dados do dataset
dataset = pd.read_csv("dataset.csv")
print(dataset.head())

# Convertendo string float através de representação
labelEncoder = LabelEncoder()
labelEncoder.fit(dataset['nome_item'])
dataset['nome_item'] = labelEncoder.transform(dataset['nome_item'])

labelEncoder.fit(dataset['horario_pedido'])
dataset['horario_pedido'] = labelEncoder.transform(dataset['horario_pedido'])

labelEncoder.fit(dataset['id_transacao'])
dataset['id_transacao'] = labelEncoder.transform(dataset['id_transacao'])

print(dataset.head())

# Método fit() executa o algoritmo e agrupar os dados
kmeans.fit(dataset)

print(kmeans.cluster_centers_)

# distance: A tabela de distâncias é criada de forma que em cada instância contém os valores de distância em relação a cada cluster
distance = kmeans.fit_transform(dataset)
print(distance)

# labels: o atributo labels_ que nos retorna os labels para cada instância, ou seja, o código do cluster que a instância de dados foi atribuído
labels = kmeans.labels_
print(labels)

# Método Elbow
wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='random')
    kmeans.fit(dataset)
    print i, kmeans.inertia_
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('O Metodo Elbow')
plt.xlabel('Numero de Clusters')
plt.ylabel('WSS')  # within cluster sum of squares
#plt.show()

#
print(kmeans.predict(dataset))

#
plt.scatter(dataset.iloc[:, 0], dataset.iloc[:,1], s = 100, c = kmeans.labels_)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'red',label = 'Centroids')
plt.title('Clusters and Centroids')
plt.xlabel('SepalLength')
plt.ylabel('SepalWidth')
plt.legend()

plt.show()