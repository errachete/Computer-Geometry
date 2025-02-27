# -*- coding: utf-8 -*-
'''
Práctica 4

Rubén Ruperto Díaz y Rafael Herrera Troca
'''

import os
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from scipy.io import netcdf as nc
from sklearn.decomposition import PCA
from matplotlib import cm

# Vamos al directorio de trabajo
os.chdir("./resources")

# Dados dos elementos y los pesos a utilizar, devuelve la 2-distancia
# entre ambos
def dist(x1, x2, w):
    return np.sqrt(np.sum(w*abs(x1-x2)**2))


## Ejercicio 1:

# Leemos el fichero correspondiente
f = nc.netcdf_file('hgt.2019.nc', 'r')
time = f.variables['time'][:].copy()
level = f.variables['level'][:].copy()
lats = f.variables['lat'][:].copy()
lons = f.variables['lon'][:].copy()
hgt_2019 = f.variables['hgt'][:].copy()
f.close()

# Hacemos PCA para obtener cuatro componentes principales
num_comp = 4
indp = (level == 500)
X = hgt_2019[:,indp,:,:]
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
plt.savefig("PCA.png")
plt.show()

# Mostramos la varianza explicada
print("La varianza explicada es:", pca.explained_variance_ratio_)


## Ejercicio 2:

# Calculamos tres arrays de booleanos según si la longitud y la latitud 
# están en el intervalo que queremos y si la presión es 1000 o 500 hPa
indx = (lons > 340) | (lons < 20)
indy = (lats > 30) & (lats < 50)
indp = (level == 1000) | (level == 500)

# Restringimos la matriz de datos a los que tienen la latitud adecuada, 
# sobre el resultado, restringimos nuevamente a los que tienen la longitud
# adecuada y, finalmente, volvemos a restringir para quedarnos sólo con los
# que tienen la presión que queremos
hgt_rest = hgt_2019[:,:,indy,:][:,:,:,indx][:,indp,:,:]

# Leemos los datos correspondientes a las alturas en 2020
f = nc.netcdf_file('hgt.2020.nc', 'r')
time_2020 = f.variables['time'][:].copy()
hgt_2020 = f.variables['hgt'][:].copy()
f.close()

# Encontramos el día 20-1-2020 y sacamos su información
a0 = (dt.date(2020, 1, 20) - dt.date(1800, 1, 1)).days*24
indt = (time_2020 == a0)
dia = hgt_2020[indt,:,:,:][:,:,indy,:][:,:,:,indx][:,indp,:,:]
dia = dia.astype(np.int32)

# Buscamos los cuatro días más parecidos al que queremos usando la distancia
# que hemos definido en la función dist
hgt_ord = sorted(hgt_rest, key=lambda x: dist(x, dia, 0.5))
time_ord = sorted(time, key=lambda x: dist(hgt_rest[time == x], dia, 0.5))
hgt_sim = hgt_ord[0:4]
time_sim = time_ord[0:4]
dias_sim = [dt.date(1800, 1, 1) + dt.timedelta(hours=t) for t in time_sim]
print("Los cuatro días más parecidos al 20-01-2020 son:")
for l in dias_sim: 
    print(l)
    
# Leemos los datos correspondientes a las temperaturas de 2019
f = nc.netcdf_file('air.2019.nc', 'r')
air_2019 = f.variables['air'][:].copy()
sc_f = f.variables['air'].scale_factor
offs = f.variables['air'].add_offset
f.close()

# Calculamos nuestra estimación de la temperatura como la media de la de los
# cuatro días análogos para toda la malla
indp = (level == 1000)
temp_sim = air_2019[[True if x in time_sim else False for x in time],indp,:,:]
temp_est = np.mean(temp_sim, axis=0)

# Leemos los datos correspondientes a las temperaturas de 2020
f = nc.netcdf_file('air.2020.nc', 'r')
air_2020 = f.variables['air'][:].copy()
f.close()

# Nos quedamos con las temperaturas estimadas y reales de la región pedida,
# con longitud y latitud en los intervalos que usamos antes
temp_real = air_2020[indt,indp,:,:]
temp_real_rest = temp_real[:,indy,:][:,:,indx]
temp_est_rest = temp_est[indy,:][:,indx]

# Calculamos el error de nuestra estimación de la temperatura con respecto a
# los datos reales y lo escalamos de acuerdo a la forma en que están
# escaladas las temperaturas en el fichero
error = np.mean(abs(temp_real_rest-temp_est_rest))
error = error*sc_f
print("El error medio cometido al estimar la temperatura en la región pedida es de", error)


# Representamos sobre dos mapas de la región restringida la temperatura 
# real y la estimada

# Es necesario tener instalada la libería de Python Basemap para ejecutar
# el código que viene a continuación

from mpl_toolkits.basemap import Basemap

# Creamos un mapa de la región en la que estamos trabajando, 
# especificando los límites de latitud y longitud
m = Basemap(projection='mill', resolution='l', llcrnrlat=32.5, urcrnrlat=47.5, llcrnrlon=-17.5, urcrnrlon=17.5, )

# Reorganizamos los datos, ya que la librería requiere que las longitudes
# estén entre -180 y 180
lons_mod = [i if i < 180 else i - 360 for i in lons[indx]]
ordlons = np.append(lons_mod[8:],lons_mod[:8])

# Hacemos una malla para posteriormente representar los datos con contourf
lon2, lat2 = np.meshgrid(ordlons,lats[indy])
x, y = m(lon2, lat2)

# Representamos el mapa con la temperatura real el dia 20/01/20, pasando
# los datos a grados Celsius
fig = plt.figure()
data = temp_real_rest[0]
data = data * sc_f + offs - 273.15
orddata = np.array([np.append(i[8:],i[:8]) for i in data])
m.drawcoastlines()
a = m.contourf(x,y,orddata,np.linspace(-1,18,20),cmap=plt.cm.get_cmap('jet'))
fig.colorbar(a)
m.shadedrelief()
plt.savefig('mapa_temp_real.png')

# Representamos el mapa con nuestra predicción para el dia 20/01/20
fig = plt.figure()
data = temp_est_rest
data = data * sc_f + offs - 273.15
orddata = np.array([np.append(i[8:],i[:8]) for i in data])
m.drawcoastlines()
a = m.contourf(x,y,orddata,np.linspace(-1,18,20),cmap=plt.cm.get_cmap('jet'))
fig.colorbar(a)
m.shadedrelief()
plt.savefig('mapa_temp_est.png')
