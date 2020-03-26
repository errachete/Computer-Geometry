# -*- coding: utf-8 -*-
'''
Práctica 5

Rubén Ruperto Díaz y Rafael Herrera Troca
'''

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d

# Vamos al directorio de trabajo
os.chdir("./resources")


## Ejercicio 1

# Obtenemos los valores requeridos de latitud y longitud
u = np.linspace(0, np.pi, 25)
v = np.linspace(0, 2 * np.pi, 50)

# Transformamos las coordenadas polares a cartesianas
x = np.outer(np.sin(u), np.sin(v))
y = np.outer(np.sin(u), np.cos(v))
z = np.outer(np.cos(u), np.ones_like(v))

# Escribimos los puntos de la curva
t2 = np.linspace(0, 1, 20000, endpoint=True)
x2 = np.sin(np.pi*t2 + np.pi/2)
y2 = np.sin(27 * np.pi*t2 + np.pi/2) * np.cos(np.pi*t2 + np.pi/2)
z2 = np.cos(27 * np.pi*t2 + np.pi/2) * np.cos(np.pi*t2 + np.pi/2)

# Representamos la esfera y la curva
plt.figure(figsize=(10,10))
ax = plt.axes(projection='3d')
ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap=cm.viridis, edgecolor='none')
ax.plot(x2, y2, z2, '-', c='w', zorder=3)
ax.auto_scale_xyz([-1, 1], [-1, 1], [-1, 1])
plt.title('Esfera con curva')
plt.savefig('sphere.png')
plt.show()
