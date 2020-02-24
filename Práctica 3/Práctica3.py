# -*- coding: utf-8 -*-
'''
Práctica 3

Rubén Ruperto Díaz y Rafael Herrera Troca
'''

from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.datasets import make_blobs
import os
import numpy as np
import matplotlib.pyplot as plt

# Carpeta donde se encuentran los archivos
ubica = "./temp"

# Vamos al directorio de trabajo
os.chdir(ubica)


## Parte 1: K-Means

# Definimos un sistema X de 1000 elementos de dos estados
# construido a partir de una muestra aleatoria entorno a unos centros
centers = [[1, 1], [-1, -1], [1, -1]]
X, true_labels = make_blobs(n_samples=1000, centers=centers, cluster_std=0.4, random_state=0)

# Representamos la muestra obtenida
plt.figure(figsize=(10,10))
plt.scatter(X[:,0], X[:,1], c='red', s=20)
plt.title("Muestra de puntos generados aleatoriamente para aplicar k-means")
plt.savefig("KMeansData.png")
plt.show()

# Los clasificamos mediante el algoritmo K-Means probando todos los valores
# de K entre 1 y 15 y calculamos el coeficiente de Silhouette para cada K
sil = np.array([])
for n_clusters in range(2,16):

    # Usamos la inicialización aleatoria "random_state=0" 
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
    labels = kmeans.labels_
    kcenters = kmeans.cluster_centers_
    silhouette = metrics.silhouette_score(X, labels)
    sil = np.append(sil, silhouette)
    colors = ['magenta', 'blue', 'lime', 'red', 'silver', 'darkorange', 'gold', 'pink', 'cyan', 'green', 'rosybrown', 'sienna', 'lightcyan', 'yellow', 'deepskyblue']
    colorlabels = [colors[i] for i in labels]
    
    plt.figure(figsize=(10,10))
    plt.scatter(X[:,0], X[:,1], c=colorlabels, s=20)
    plt.scatter(kcenters[:,0], kcenters[:,1], marker='x', s=100, c='black')
    plt.title("Clusters con $K=$" + str(n_clusters))
    plt.savefig("KMeans" + str(n_clusters) + ".png")
    plt.show()
    print("Silhouette Coefficient: %0.3f" % silhouette)

# Representamos en un gráfico el valor del coeficiente de silhouette para
# cada K
plt.figure(figsize=(10,10))
plt.plot(range(2,16), sil, 'ko-')
plt.title(r"Valor de $\bar{s}$ según $K$")
plt.xlabel("$K$")
plt.ylabel(r"$\bar{s}$")
plt.savefig("KMeansSilh.png")
plt.show()


## Parte 2: DBSCAN

