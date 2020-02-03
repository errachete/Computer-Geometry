# -*- coding: utf-8 -*-
"""
Práctica 1

Rubén Ruperto Díaz y Rafael Herrera Troca
"""


def f(x):
    return r*x*(1-x);

def fn(x0,fun,n):
    x = x0;
    for i in range(n):
        x = fun(x);
    return x;

def orb(x0,fun,long):
    o = [x0];
    for i in range(long):
        o.append(fun(o[-1]));
    return o;

def limite(suc,eps,rango):
    last = suc[-1*(range(rango)+1)];
    # Nos hemos quedado aqui

# Calculamos la órbita parcial para r = r
    
r = 3.1,
x0 = 0;
orb_n = orb(x0,f,200);

#ploteamos la orbita para ser guays

# Calculamos V0

