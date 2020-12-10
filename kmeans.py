# This Python file uses the following encoding: utf-8
import pandas as pd
from sklearn.cluster import KMeans

# 3 cluster: horario pedido, localidade, nome_item
# random: Se refere ao modo de inicialização de forma aleatória,
#    ou seja, os centróides iniciais serão gerados de forma totalmente aleatória sem um critério para seleção
kmeans = KMeans(n_clusters = 3, init = 'random')

# Recebendo os dados do dataset
dataset = pd.read_csv("dataset.csv")
print(dataset.head())

# Método fit() executa o algoritmo e agrupar os dados
kmeans.fit(dataset)