# -*- coding: utf-8 -*-
'''
Práctica 5

Rubén Ruperto Díaz y Rafael Herrera Troca
'''

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import animation
from mpl_toolkits.mplot3d import axes3d

# Vamos al directorio de trabajo
os.chdir("./resources")


# Proyección estereográfica: Dado un punto de R^3, devuelve el punto
# de R^2 en el que se transforma (sumergido en R^3 poniendo z = 0)
# Para evitar la división entre 0, añadimos un épsilon de 10^-20
# al divisor
def proy_est(x, y, z):
    eps = 10**-20
    xp = x / (1 + z + eps)
    yp = y / (1 + z + eps)
    zp = np.zeros_like(z)
    return xp, yp, zp

# Dadas unas coordenadas cartesianas x, y, z pertenecientes a la
# esfera unidad, las transforma en las coordenadas esféricas
# correspondientes
def cart2esf(x, y, z):
    
    c1 = np.array(z > 0)
    c2 = np.array(z < 0)
    c3 = np.array(z == 0)
    
    phi = np.zeros_like(z)
    
    phi[c1] = np.arctan(np.sqrt(x[c1]**2+y[c1]**2)/z[c1])
    phi[c2] = np.pi + np.arctan(np.sqrt(x[c2]**2+y[c2]**2)/z[c2])
    phi[c3] = np.pi / 2
        
    c1 = np.array(x > 0) & np.array(y >= 0)
    c2 = np.array(x > 0) & np.array(y < 0)
    c3 = np.array(x == 0) & np.array(y >= 0)
    c4 = np.array(x == 0) & np.array(y < 0)
    c5 = np.array(x < 0)

    theta = np.zeros_like(x)    

    theta[c1] = np.arctan(y[c1]/x[c1])
    theta[c2] = 2*np.pi + np.arctan(y[c2]/x[c2])
    theta[c3] = np.pi/2
    theta[c4] = -np.pi/2
    theta[c5] = np.pi + np.arctan(y[c5]/x[c5])
        
    return phi, theta

# Familia paramétrica dada en el ejercicio 2
def param2(t, x, y, z):
    eps = 10**-20
    xt = x / ((1-t) + abs(-1 - z)*t + eps)
    yt = y / ((1-t) + abs(-1 - z)*t + eps)
    zt = -t + z*(1-t)
    
    return xt, yt, zt
    
# Familia paramétrica definida para el ejercicio 3
# Avanza desde la identidad hasta la proyección estereográfica de
# coordenadas polares a cartesianas según el valor de t
def param3(t, phi, theta, z):
    
    # Aplicamos la transformación según t
    xt = np.cos(theta) * (t*np.tan(t*phi/2 + (1-t)*np.arctan(phi/2)) + (1-t)*np.sin(phi))
    yt = np.sin(theta) * (t*np.tan(t*phi/2 + (1-t)*np.arctan(phi/2)) + (1-t)*np.sin(phi))
    zt = -t + z*(1-t)
    
    return xt, yt, zt

# Función para animar la deformación definida por la familia
# paramétrica dada por parámetro en cada tiempo t
def animate(t, fam, x, y, z, x2, y2, z2, fix):
    xt, yt, zt = fam(t, x, y, z)
    x2t, y2t, z2t = fam(t, x2, y2, z2)
    
    ax = plt.axes(projection='3d')
    ax.set_zlim3d(-1,1)
    ax.plot_surface(xt, yt, zt, rstride=1, cstride=1, cmap=cm.viridis, edgecolor='none')
    ax.plot(x2t, y2t, z2t, '-', c='w', zorder=3)
    if fix:
        ax.auto_scale_xyz([-1, 1], [-1, 1], [-1, 1])
    
    return ax


## Ejercicio 1

# Obtenemos los valores requeridos de latitud y longitud
u = np.linspace(0, np.pi, 25)
v = np.linspace(0, 2 * np.pi, 50)

# Transformamos las coordenadas polares a cartesianas
x = np.outer(np.sin(u), np.cos(v))
y = np.outer(np.sin(u), np.sin(v))
z = np.outer(np.cos(u), np.ones_like(v))

# Escribimos los puntos de la curva
t2 = np.linspace(0, 1, 20000)
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

# Proyectamos la esfera
xp, yp, zp = proy_est(x, y, z)

# Proyectamos la curva
x2p, y2p, z2p = proy_est(x2, y2, z2)

# Representamos la proyección de la esfera y la curva
plt.figure(figsize=(10,10))
ax = plt.axes(projection='3d')
ax.plot_surface(xp, yp, zp, rstride=1, cstride=1, cmap=cm.viridis, edgecolor='none')
ax.plot(x2p, y2p, z2p, '-', c='w', zorder=3)
ax.set_zlim3d(-1,1)
plt.title('Proyección de la esfera con curva')
plt.savefig('sphere_proy.png')
plt.show()


## Ejercicio 2

# Usamos la función animate definida arriba para crear la 
# animación de la familia paramétrica dada en param2
tvalues = np.linspace(0, 1, 150, endpoint=True)
fig = plt.figure(figsize=(10,10))
ani = animation.FuncAnimation(fig, animate, tvalues, fargs=(param2, x, y, z, x2, y2, z2, False))
ani.save('ani_ej2.gif', writer='imagemagick', fps= 30)

fig = plt.figure(figsize=(10,10))
ani = animation.FuncAnimation(fig, animate, tvalues, fargs=(param2, x, y, z, x2, y2, z2, True))
ani.save('ani_ej2_fix.gif', writer='imagemagick', fps= 30)


## Ejercicio 3 (voluntario)

# Pasamos las coordenadas cartesianas a esféricas porque nuestra
# parametrización utiliza estas coordenadas
phi, theta = cart2esf(x, y, z)
phi2, theta2 = cart2esf(x2, y2, z2)

# Usamos la función animate definida arriba para crear la 
# animación de la familia paramétrica dada en param3
tvalues = np.linspace(0, 1, 100, endpoint=True)
fig = plt.figure(figsize=(10,10))
ani = animation.FuncAnimation(fig, animate, tvalues, fargs=(param3, phi, theta, z, phi2, theta2, z2, False))
ani.save('ani_ej3.gif', writer='imagemagick', fps=20)

fig = plt.figure(figsize=(10,10))
ani = animation.FuncAnimation(fig, animate, tvalues, fargs=(param3, phi, theta, z, phi2, theta2, z2, True))
ani.save('ani_ej3_fix.gif', writer='imagemagick', fps=20)