# -*- coding: utf-8 -*-
'''
Práctica 3

Rubén Ruperto Díaz y Rafael Herrera Troca
'''

from sklearn.cluster import KMeans, DBSCAN
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
colors = ['magenta', 'blue', 'lime', 'red', 'silver', 'darkorange', 'gold', 'pink', 'cyan', 'green', 'rosybrown', 'sienna', 'lightcyan', 'yellow', 'deepskyblue']
X, true_labels = make_blobs(n_samples=1000, centers=centers, cluster_std=0.4, random_state=0)

# Representamos la muestra obtenida con los clusters reales
colorlabels = [colors[i] for i in true_labels]
plt.figure(figsize=(10,10))
plt.scatter(X[:,0], X[:,1], c=colorlabels, s=20)
plt.title("Muestra de puntos generados aleatoriamente para aplicar algoritmos de clustering")
plt.savefig("Data.png")
plt.show()

# Los clasificamos mediante el algoritmo K-Means probando todos los valores
# de K entre 1 y 15 y calculamos el coeficiente de Silhouette para cada K
sil = np.array([])
for n_clusters in range(1,16):

    # Usamos la inicialización aleatoria "random_state=0" 
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
    labels = kmeans.labels_
    kcenters = kmeans.cluster_centers_
    silhouette = metrics.silhouette_score(X, labels) if len(set(labels)) != 1 else -1
    sil = np.append(sil, silhouette)
    colorlabels = [colors[i] for i in labels]
    
    plt.figure(figsize=(10,10))
    plt.scatter(X[:,0], X[:,1], c=colorlabels, s=20)
    plt.scatter(kcenters[:,0], kcenters[:,1], marker='x', s=100, c='black')
    plt.title("Clusters con $K=$" + str(n_clusters))
    plt.savefig("KMeans" + str(n_clusters) + ".png")
    plt.show()
    print("Coeficiente de Silhouette:", silhouette)

# Representamos en un gráfico el valor del coeficiente de silhouette para
# cada K
plt.figure(figsize=(10,10))
plt.plot(range(1,16), sil, 'ko-')
plt.title(r"Valor de $\bar{s}$ según $K$")
plt.xlabel("$K$")
plt.ylabel(r"$\bar{s}$")
plt.savefig("KMeansSilh.png")
plt.show()


## Parte 2: DBSCAN

# Los clasificamos mediante el algoritmo DBSCAN con épsilon en el
# itervalo [0.1, 0.4] variando de 0.05 en 0.05 y calculamos el coeficiente
# de Silhouette para cada épsilon
sil = [np.array([]),np.array([])]
X = np.array(X)
colors.append('black')
metrica = ['euclidean', 'manhattan']
for epsilon in np.arange(0.1, 0.45, 0.05):
    for j in range(0,2):
        db = DBSCAN(eps=epsilon, min_samples=10, metric=metrica[j]).fit(X)
        labels = np.array(db.labels_)
        silhouette = metrics.silhouette_score(X, labels) if len(set(labels)) != 1 else -1
        sil[j] = np.append(sil[j], silhouette)
        colorlabels = [colors[i] for i in labels]
        
        plt.figure(figsize=(10,10))
        plt.scatter(X[:,0], X[:,1], c=colorlabels, s=20)
        plt.title(r"Clusters con $\epsilon=$" + str(round(epsilon,2)))
        plt.savefig("DBSCAN"+ str(metrica[j]) + str(round(epsilon,2)) + ".png")
        plt.show()
        print("Coeficiente de Silhouette:", silhouette)

# Representamos en un gráfico el valor del coeficiente de silhouette para
# cada épsilon
plt.figure(figsize=(10,10))
plt.plot(np.arange(0.1, 0.45, 0.05), sil[0], 'ko-', label="Métrica euclidea")
plt.plot(np.arange(0.1, 0.45, 0.05), sil[1], 'ro-', label="Métrica manhattan")
plt.title(r"Valor de $\bar{s}$ según $\epsilon$")
plt.xlabel(r"$\epsilon$")
plt.ylabel(r"$\bar{s}$")
plt.legend(loc="lower right")
plt.savefig("DBSCANSilh.png")
plt.show()