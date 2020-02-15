# -*- coding: utf-8 -*-
'''
Práctica 1

Rubén Ruperto Díaz y Rafael Herrera Troca
'''

import os
import numpy as np
import matplotlib.pyplot as plt

os.chdir("D:/Universidad/Geometría Computacional/Prácticas/Práctica 1");

# Función logística f utilizada para construir el sistema dinámico
def f(x,r):
    return r*x*(1-x)

# Cálculo del término x_n = f^n(x_0)
def fn(x0,r,fun,n):
    x = x0
    for i in range(n):
        x = fun(x,r)
    return x

# Cálculo de la órbita con "long" elementos para con una función y un x_0 dados
def orb(x0,r,fun,long):
    o = [x0]
    for i in range(long):
        o.append(fun(o[-1],r))
    return o

# Cálculo del límite de una sucesión teniendo en cuenta los últimos "rango" términos
# y para un cierto épsilon
def limite(suc,eps,rango):
    last = suc[-1:-1*rango-1:-1]
    lim = [last[0]]
    per = 1
    while per < rango and abs(last[0] - last[per]) >= eps:
        lim.append(last[per])
        per = per+1
    return lim

# Compara si dos conjuntos son iguales con un error de un cierto epsilon dado
def iguales(c1, c2, eps):
    c1.sort()
    c2.sort()
    if len(c1) != len(c2):
        return False
    else:
        for i in range(len(c1)):
            if abs(c1[i] - c2[i]) >= eps:
                return False
        return True

# Calcula una estimación del error cometido al calcular un cierto V_0 dado
def error(V0,fun,r):
    V = [np.array(V0)]
    E = []
    for i in range(11):
        V.append(np.array([fun(v0,r) for v0 in V[i]]))
        V[i+1].sort()
        E.append(max(V[i+1] - V[i]))
    return E[0]
    

# Calculamos la órbita parcial para un cierto conjunto de r y distintos x_0.
# Para r en {3, 3.1, 3.2, 3.3, 3.4, 3.5}, 
# dibujamos las gráficas para los primeros 50 elementos de la órbita.
# Una vez calculada cada órbita, hacemos el límite para cada x_0 para obtener 
# los conjuntos atractores V_0 y comprobamos si son o no conjuntos atractores
# según el valor del periodo que tengan y, en caso de que lo sean, 
# que coinciden para los distintos valores de x_0. 
# Finalmente, almacenamos V_0 para cada r.

epsilon = 10**(-5)
x0 = [0.25, 0.5, 0.75]
rValues = np.arange(0.1,4,0.01)
V0 = []
for r in rValues:
    r = round(r,2)
    # Cálculo de la órbita
    orb_r = [orb(x,r,f,10000) for x in x0]
    # Representación de la órbita (si procede)
    if r >= 3 and r <= 3.5 and r == round(r,1):
        plt.figure(figsize=(10,10))
        plt.plot(orb_r[0][:50], 'r', label="$x_0 = 0.25$")
        plt.plot(orb_r[1][:50], 'g', label="$x_0 = 0.5$")
        plt.plot(orb_r[2][:50], 'b', label="$x_0 = 0.75$")
        plt.title("Órbita en intervalo [0-50] para $r = $" + str(r))
        plt.gca().set_ylim([0,1])
        plt.legend(loc="lower right")
        plt.savefig("orb_r" + str(r) + ".png")
        plt.show()
    # Cálculo de los conjuntos atractores y su periodo
    V0_r = [limite(orbi,epsilon,100) for orbi in orb_r]
    # Si el periodo es igual a los 100 elementos que hemos consultado, consideramos
    # que no hay conjunto atractor
    if len(V0_r[0]) == 100:
        print("Para r =", r, " no existe conjunto atractor.")
    # Si hay un periodo menor, comprobamos que los conjuntos atractores para cada
    # x_0 coinciden (con diferencia menor que epsilon) y lo mostramos
    else:
        for i in range(len(V0_r)-1):
            if not iguales(V0_r[i], V0_r[i+1], epsilon):
                print("Para r =", r, ", los conjuntos atractores para x0 =", x0[i],
                      "y x0 =", x0[i+1], "difieren en más de", epsilon)
        print("Para r =", r, " el conjunto atractor es V0 =", V0_r[0])
    # Guardamos el conjunto V_0 obtenido (sea o no atractor)
    V0_r[0].sort()
    V0.append(V0_r[0])
        
    
# Una vez tenemos todos los conjuntos V_0 para cada valor, los representamos
# en una gráfica en función de r.
V0_ = []
r_ = []
for i in range(len(V0)):
    for j in range(len(V0[i])):
        V0_.append(V0[i][j])
        r_.append(rValues[i])
plt.figure(figsize=(10,10))
plt.scatter(r_, V0_, s=0.5)
plt.title("Valores en $V_0$ en funcion de $r$")
plt.xlabel("Valor de $r$")
plt.ylabel("Valores en $V_0$")
plt.savefig("V0Values.png")
plt.show()

# Calculamos el menor r de los que hemos considerado para el cual el conjunto atractor
# tiene 8 elementos.
i = 0
while len(V0[i]) != 8:
    i = i + 1
print("El menor r de los considerados tal que V0 tiene 8 elementos es " + str(round(rValues[i],2)) + ".")
print("El conjunto atractor correspondiente a dicha r es:", V0[i])

# Estimamos el error cometido al calcular V_0 para cada valor de r y lo representamos
# en una gráfica
E = [error(V0[k],f,round(rValues[k],2)) for k in range(len(rValues))]
plt.figure(figsize=(10,10))
plt.plot(rValues,E)
plt.title("Error en $V_0$ en funcion de $r$")
plt.xlabel("Valor de $r$")
plt.ylabel("Error en $V_0$")
plt.savefig("V0Error.png")
plt.show()

#Calculamos el error para el V_0 obtenido en el apartado anterior (el primero con 8 elementos)
print("El error al calcular el conjunto atractor correspondiente a r = " + str(round(rValues[i],2)) + 
      " es de " + str(E[i]) + ".")


    

    

