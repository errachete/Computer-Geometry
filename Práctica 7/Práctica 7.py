# -*- coding: utf-8 -*-
'''
Práctica 7

Rubén Ruperto Díaz y Rafael Herrera Troca
'''

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import animation
from skimage import io,color

# Vamos al directorio de trabajo
os.chdir("./resources")


# Calcula el diámetro de un conjunto dado con coordenadas
# en las matrices x, y, z
def diametro(x, y, z, paso=1):
    diam = 0
    puntos = np.array(list(zip(x.ravel(),y.ravel(),z.ravel())))
    for i in range(0, len(puntos), paso):
        for j in range(i, len(puntos), paso):
            d = np.sqrt(np.sum((puntos[i]-puntos[j])**2))
            diam = d if d > diam else diam
    return diam

# Calcula el centroide de un conjunto dado con coordenadas
# en las matrices x, y, z
def centroide(x, y, z):
    cx = np.sum(x) / len(x.ravel())
    cy = np.sum(y) / len(y.ravel())
    cz = np.sum(z) / len(z.ravel())
    return cx, cy, cz

# Dado un conjunto de coordenadas x, y, z en matrices de
# cualquier forma, aplica la transformación afín resultante
# de aplicar una rotación sobre el punto dado en c con la matriz
# M y una traslación con el vector v y devuelve las coordenadas
# transformadas con la misma forma que fueron dadas
def transf(x, y, z, c, M, v=np.array([0,0,0])):
    X = x.copy().ravel()
    Y = y.copy().ravel()
    Z = z.copy().ravel()
    for i in range(len(X)):
        q = np.array([X[i]-c[0],Y[i]-c[1],Z[i]-c[2]])
        X[i], Y[i], Z[i] = np.matmul(M, q) + np.array(list(c)) + v
    X = np.reshape(X, np.shape(x))
    Y = np.reshape(Y, np.shape(y))
    Z = np.reshape(Z, np.shape(z))
    return X, Y, Z

# Función que define la familia paramétrica pedida en la
# práctica (rotar 3pi y trasladar el diámetro de la figura,
# que viene dado en d) para cada valor de t
def param(t, x, y, z, c, d):
    Mt = np.array([[np.cos(3*np.pi*t), -np.sin(3*np.pi*t), 0],
                   [np.sin(3*np.pi*t), np.cos(3*np.pi*t),  0],
                   [0,                 0,                  1]])
    vt = np.array([d*t, d*t, 0])
    return transf(x,y,z,c,Mt,vt)
    
# Función para animar la deformación definida por la familia
# paramétrica dada por parámetro en cada tiempo t sobre los puntos
# x,y,z dados
def animate1(t, fam, x, y, z, c, d):
    xt, yt, zt = fam(t, x, y, z, c, d)
    
    ax = plt.axes(projection='3d')
    ax.plot_surface(xt,yt,zt,cmap=plt.cm.get_cmap('viridis'))
    ax.auto_scale_xyz([-4, 13], [-4, 13], [-1, 1])
    
    return ax

# Función para animar la deformación definida por la familia
# paramétrica dada por parámetro en cada tiempo t sobre los puntos
# x,y,z y con los colores dados en formato RGBA en col
def animate2(t, fam, x, y, z, col, c, d):
    xt, yt, zt = fam(t, x, y, z, c, d)
    
    ax = plt.axes(projection='3d')
    ax.scatter(xt,yt,zt,c=col,s=1,marker='.')
    ax.auto_scale_xyz([0, 800], [0, 800], [-1, 1])
    
    return ax


## Ejercicio 1

# Comenzamos por construir la figura que vamos a utilizar
x = np.linspace(-np.pi, np.pi, 51)
y = np.linspace(-np.pi, np.pi, 51)
x,y = np.meshgrid(x,y)
z = np.sin(x)*np.sin(y)

# Representamos la figura
plt.figure(figsize=(10,10))
ax = plt.axes(projection='3d')
ax.plot_surface(x,y,z,cmap=plt.cm.get_cmap('viridis'))
plt.title('Figura utilizada en el ejercicio 1')
plt.savefig('fig.png')
plt.show()

# Calculamos su diámetro y su centroide
d = diametro(x,y,z)
c = centroide(x,y,z)

# Construimos la animación
tvalues = np.linspace(0, 1, 150, endpoint=True)
fig = plt.figure(figsize=(10,5))
ani = animation.FuncAnimation(fig, animate1, tvalues, fargs=(param, x, y, z, c, d))
ani.save('ani_ej1.gif', writer='imagemagick', fps=30)


## Ejercicio 2

# Leemos la imagen, construimos la malla para representarlo y
# nos quedamos sólo con las posiciones con rojo < 240
img = io.imread('arbol.png')
x = np.arange(np.shape(img)[0])
y = np.arange(np.shape(img)[1])
x,y = np.meshgrid(x,y)
z = np.zeros_like(x)

filtro = img[:,:,0] < 240
x = x[filtro]
y = y[filtro]
z = z[filtro]
col = img[filtro] / 256

# Representamos el subsistema
plt.figure(figsize=(10,10))
ax = plt.axes(projection='3d')
ax.scatter(x,y,z,c=col,s=1,marker='.')
plt.title('Figura utilizada en el ejercicio 2')
plt.savefig('fig.png')
plt.show()

# Calculamos su diámetro y su centroide
d = diametro(x,y,z,10)
c = centroide(x,y,z)
print("El centroide es", c)

# Construimos la animación
tvalues = np.linspace(0, 1, 150, endpoint=True)
fig = plt.figure(figsize=(10,10))
ani = animation.FuncAnimation(fig, animate2, tvalues, fargs=(param, x, y, z, col, c, d))
ani.save('ani_ej2.gif', writer='imagemagick', fps=30)

