# This Python file uses the following encoding: utf-8
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder

from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import fcluster
from scipy.cluster.hierarchy import dendrogram

from matplotlib import pyplot
import cufflinks as cf

cf.go_offline()
cf.set_config_file(offline=False, world_readable=True)

print("Algoritmos de Clustering: K-means\n")

# 5 cluster: horario pedido, localidade, nome_item
# random: Se refere ao modo de inicialização de forma aleatória,
#    ou seja, os centróides iniciais serão gerados de forma totalmente aleatória sem um critério para seleção
kmeans = KMeans(n_clusters = 5, init = 'random')

# Recebendo os dados do dataset
dataset = pd.read_csv("dataset.csv")
print("Os 5 primeiras linhas:")
print(dataset.head())

print("\nVenda Acumulativa")
data_nome_item = dataset.loc[:, ['nome_item']]
print(data_nome_item)

#data_nome_item["nome_item"].iplot(kind="bar")

data_nome_item.nome_item.value_counts().iplot(kind='bar', title='Status dos pedidos')

data_parcial = dataset.iloc[:,3:5]
print(data_parcial.groupby(['nome_item'])['quantidade_item'].agg('sum'))
#print(dataset.groupby(by=['Fruit','Date']).sum().groupby(level=[0]).cumsum())
#print(dataset.groupby(by=['nome_item','quantidade_item']).sum())

# Representação dos dados:
#plt.bar(dataset['nome_item'],(dataset['quantidade_item'].sum()))
#plt.show()

# Convertendo string em float através de representação
#   por exemplo, um grupo de item=pizza representa um número, assim, pode ser contabilizado pelo algoritmo, em vez de trabalhar com string tabalha com número
labelEncoder = LabelEncoder()
labelEncoder.fit(dataset['nome_item'])
dataset['nome_item'] = labelEncoder.transform(dataset['nome_item'])

labelEncoder.fit(dataset['horario_pedido'])
dataset['horario_pedido'] = labelEncoder.transform(dataset['horario_pedido'])

labelEncoder.fit(dataset['id_transacao'])
dataset['id_transacao'] = labelEncoder.transform(dataset['id_transacao'])

print("\n Os 5 primeiras linhas, depois da representação:")
print(dataset.head())


print("\n Dendrograma")
dados = dataset.values
h = linkage(dados, method='complete', metric='euclidean')
dendrogram(h)
pyplot.show()
#rotulos_dist = fcluster(h, t=7.5, criterion='distance')
#rotulos_k = fcluster(h, t=3, criterion='maxclust')

# Método fit() executa o algoritmo e agrupar os dados
print("\n Configuração do algoritmo:")
print(kmeans.fit(dataset))

print("\n Imprimi os clustering ou centroídes gerados:")
data = kmeans.cluster_centers_
print(data)

print("\n O centroide que apresentar a menor distância, será o cluster escolhido:")
print(kmeans.predict(data))

# distance: A tabela de distâncias é criada de forma que em cada instância contém os valores de distância em relação a cada cluster
distance = kmeans.fit_transform(dataset)
print("\n Tabela de distâncias:")
print(distance)

# labels: o atributo labels_ que nos retorna os labels para cada instância, ou seja, o código do cluster que a instância de dados foi atribuído
labels = kmeans.labels_
print("\n Retorna os labels de cada instâncias: ")
print(labels)

# Método Elbow
#wcss = []

# print("\n Encontrar valor ideal de k(nº de clustering ideal): ")
# for i in range(1, 11):
#     kmeans = KMeans(n_clusters=i, init='random')
#     kmeans.fit(dataset)
#     print i, kmeans.inertia_
#     wcss.append(kmeans.inertia_)
# plt.plot(range(1, 11), wcss)
# plt.title('O Metodo Elbow')
# plt.xlabel('Numero de Clusters')
# plt.ylabel('WSS')
# plt.show()

#print("\n O centroide que apresentar a menor distância, será o cluster escolhido:")
#print(kmeans.predict(data))


# Colunas:
#df.iloc[:,0] # Todos os dados da primeira coluna do dataset
#df.iloc[0:5,-1] # Do primeiro ao quinto dado da última coluna

# plt.scatter(dataset.iloc[:, 0], dataset.iloc[:,1], s = 100, c = kmeans.labels_)
# plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'red',label = 'Centroids')
# plt.title('Clusters and Centroids')
# plt.xlabel('horario_pedido')
# plt.ylabel('id_transacao')
# plt.legend()
# plt.show()