# -*- coding: utf-8 -*-
'''
Práctica 3

Rubén Ruperto Díaz y Rafael Herrera Troca
'''

import os
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from scipy.io import netcdf as nc
from sklearn.decomposition import PCA
from matplotlib import cm

# Carpeta donde se encuentran los archivos
ubica = "./temp"

# Vamos al directorio de trabajo
os.chdir(ubica)


## Ejercicio 1:

# Leemos el fichero correspondiente
f = nc.netcdf_file('hgt.2019.nc', 'r')
time = f.variables['time'][:].copy()
time_bnds = f.variables['time_bnds'][:].copy()
level = f.variables['level'][:].copy()
lats = f.variables['lat'][:].copy()
lons = f.variables['lon'][:].copy()
hgt = f.variables['hgt'][:].copy()
time_units = f.variables['time'].units
hgt_units = f.variables['hgt'].units
f.close()

# Hacemos PCA para obtener cuatro componentes principales
num_comp = 4
X = hgt[:,5,:,:]
X2D = X.reshape(len(time), len(lats)*len(lons))
pca = PCA(n_components=num_comp)
redX2D = pca.fit_transform(X2D.T).T
redX = redX2D.reshape(num_comp, len(lats), len(lons))

# Representamos como gráfica en 3 dimensiones y gráfica de curvas de nivel
# las cuatro componentes principales obtenidas
fig = plt.figure()
fig.subplots_adjust(hspace=0.4, wspace=0.4)
for i in range(1, 5):
    ax = fig.add_subplot(2, 2, i)
    ax.text(0.5, 90, 'PCA-'+str(i),
           fontsize=18, ha='center')
    plt.contour(lons, lats, redX[i-1,:,:], cmap=cm.rainbow_r)
plt.show()

# Mostramos la varianza explicada
print()



## Ejercicio 2:

