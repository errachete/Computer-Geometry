# -*- coding: utf-8 -*-
'''
Práctica 6

Rubén Ruperto Díaz y Rafael Herrera Troca
'''

import os
import numpy as np
import matplotlib.pyplot as plt

# Vamos al directorio de trabajo
os.chdir("./resources")


# Ecuación del sistema dinámico continuo
def F(q):
    ddq = -2*q * (q**2 - 1)
    return ddq

# Resolución de la ecuación dinámica q' = F(q), obteniendo la órbita q(t)
# Los valores iniciales son la posición q0 := q(0) y la derivada dq0 := q'(0)
def orbita(n, q0, dq0, F, d):
    q = np.empty([n+1])
    q[0] = q0
    q[1] = q0 + dq0*d
    for i in np.arange(2,n+1):
        args = q[i-2]
        q[i] = - q[i-2] + d**2*F(args) + 2*q[i-1]
    return q

# Cálculo de una aproximación discreta de la derivada de q(t) con
# un cierto delta 'd' pasado por parámetro
def deriv(q, dq0, d):
   dq = (q[1:len(q)] - q[0:(len(q)-1)]) / d
   dq = np.insert(dq, 0, dq0)
   return dq

# Representa en la gráfica activa el diagrama de fases para los datos 
# iniciales q0 y dq0 dados
def diag_fases(q0, dq0, F, d, n, col='red'): 
    q = orbita(n, q0, dq0, F, d)
    dq = deriv(q, dq0, d)
    p = dq/2
    plt.plot(q, p, c=col)

# Devuelve el área encerrada por el diagrama de fases para los datos
# iniciales q0 y dq0
def area(q0, dq0, F, d, n):
    q = orbita(n, q0, dq0, F, d)
    dq = deriv(q, dq0, d)
    p = dq/2
    
    w = np.arange(1, len(q)-1)
    w = w[np.where((q[w-1] > q[w]) & (q[w] < q[w+1]))]
    
    if len(w) == 0:
        return 0
    else:
        valle = np.arange(w[0], w[1])
        mitad = valle[np.where(p[valle] >= 0)]
        return 2 * np.trapz(p[mitad],q[mitad])
    
    
## Ejercicio 1

d = 10**(-4)
n = int(32/d)
seq_q0 = np.linspace(0, 1, 50)
seq_dq0 = np.linspace(2, 0, 50)

# Representamos los diagramas de fases para cada dato inicial
plt.figure(figsize=(10,10))
for i,dq0 in enumerate(seq_dq0):
    col = i * 0.5 / len(seq_dq0)
    diag_fases(0, dq0, F, d, n, plt.get_cmap("viridis")(col))
for i,q0 in enumerate(seq_q0):
    col = 0.5 + i * 0.5 / len(seq_q0)
    diag_fases(q0, 0, F, d, n, plt.get_cmap("viridis")(col))
plt.title(r"Diagrama de fases del sistema con distintos valores para $q_0$ y $\dot{q}_0$")
plt.axis('equal')
plt.xlabel(r"$q(t)$")
plt.ylabel(r"$p(t)$")
plt.savefig("fases.png")
plt.show()


## Ejercicio 2

# Calculamos el área para cada dato inicial
areas = np.empty(len(seq_dq0) + len(seq_q0))
for i,dq0 in enumerate(seq_dq0):
    areas[i] = area(0, dq0, F, d, n)
for i,q0 in enumerate(seq_q0):
    areas[i+len(seq_dq0)] = area(q0, 0, F, d, n)

# Obtenemos la mayor de las áreas y el dato inicial al que corresponde
max_area = np.max(areas)
indx = np.argmax(areas)
max_q0, max_dq0 = (0, seq_dq0[indx]) if indx < len(seq_dq0) else (seq_q0[indx-len(seq_dq0)], 0)
print("La mayor de las áreas es", max_area)
print("Se obtiene para (q0,dq0) = (" + str(max_q0) + "," + str(max_dq0) + ")")

# Obtenemos el área ínfima, correspondiente al hueco que se ve en
# la parte izquierda del gráfico con todos los diagramas de fases
# Si tomamos el diagrama correspondiente a q0 = 0 y el menor dq0 mayor 
# que 0, observamos que delimita el hueco de la izquierda y otro igual 
# a la derecha, por lo que el área ínfima sería la mitad de la de 
# este diagrama de fases
inf_q0, inf_dq0 = (0, seq_dq0[-2])
inf_area = area(inf_q0, inf_dq0, F, d, n) / 2
print("El ínfimo de las áreas es", inf_area)
print("Es la mitad de la obtenida para (q0,dq0) = (" + str(inf_q0) + "," + str(inf_dq0) + ")")
area_tot = max_area - inf_area
print("Luego el área total es", area_tot)

# Representamos sobre el gráfico con todos los diagramas de fases el
# que proporciona el área máxima y el que delimita la ínfima (con sus
# dos mitades)
plt.figure(figsize=(10,10))
for i,dq0 in enumerate(seq_dq0):
    col = i * 0.5 / len(seq_dq0)
    diag_fases(0, dq0, F, d, n, plt.get_cmap("viridis")(col))
for i,q0 in enumerate(seq_q0):
    col = 0.5 + i * 0.5 / len(seq_q0)
    diag_fases(q0, 0, F, d, n, plt.get_cmap("viridis")(col))
diag_fases(max_q0, max_dq0, F, d, n)
diag_fases(inf_q0, inf_dq0, F, d, n)
plt.title(r"Diagrama de fases del sistema y límites del área")
plt.axis('equal')
plt.xlabel(r"$q(t)$")
plt.ylabel(r"$p(t)$")
plt.savefig("areas.png")
plt.show()

# Ahora que sabemos calcular el área, podemos hacer una estimación del 
# error cometido utilizando un delta más pequeño
d2 = 10**(-5)
n2 = int(32/d2)
max_area2 = area(max_q0, max_dq0, F, d2, n2)
inf_area2 = area(inf_q0, inf_dq0, F, d2, n2) / 2
area_tot2 = max_area2 - inf_area2
print("El área total con un delta más fino es", area_tot2)
print("El error cometido en el primer cálculo del área es", abs(area_tot - area_tot2))