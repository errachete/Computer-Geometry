# -*- coding: utf-8 -*-
'''
Práctica 2

Rubén Ruperto Díaz y Rafael Herrera Troca
'''

import os
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from queue import PriorityQueue

# Carpeta donde se encuentran los archivos
ubica = "D:/Universidad/Geometría Computacional/Prácticas/Práctica 2"

# Vamos al directorio de trabajo
os.chdir(ubica)


# Clase para definir un nodo de un árbol binario.
# Se tiene el texto contenido en el nodo y sus dos hijos izquierdo
# y derecho, que son a su vez nodos de árbol binario (TreeNode).
class TreeNode:
    def __init__(self, root_a, left_a, right_a):
        self.root = root_a;
        self.left = left_a;
        self.right = right_a;

# Funciones para mostrar un árbol binario construido a base de TreeNode
def show_tree(tree):
    print(repr(tree.root))
    __show_tree(tree, '')
def __show_tree(tree, pre):
    if tree.left != None:
        print(pre + chr(9507) + chr(9473), repr(tree.left.root), sep='')
        __show_tree(tree.left, pre + chr(9475) + ' ')
    if tree.right != None:
        print(pre + chr(9495) + chr(9473), repr(tree.right.root), sep='')
        __show_tree(tree.right, pre + '  ')

# Función que calcula el siguiente nodo del árbol de Huffman y actualiza
# la tabla de codificación y la lista de probabilidades de la forma correspondiente.
# Recibe la probabilidad de aparición de cada caracter, la lista de nodos generados 
# hasta el momento y la tabla de codificación hasta el momento.
def huffman_branch(distr, treeList, table):
    # Extraemos los dos elementos con menor probabilidad de aparición
    first = distr.get()
    second = distr.get()
    # Calculamos el nuevo elemento como la concatenación de los dos extraídos
    # y su probabilidad como la suma de las probabilidades
    state_new = first[1] + second[1]
    probab_new = first[0] + second[0]
    
    # Creamos y almacenamos un nuevo nodo del árbol de Huffman poniendo como hijos izquierdo
    # y derecho los nodos correspondientes y como raíz el nuevo elemento calculado
    # en el paso anterior como concatenación de los dos extraídos
    node_new = TreeNode(state_new, treeList[first[1]], treeList[second[1]])
    treeList.pop(first[1])
    treeList.pop(second[1])
    treeList[state_new] = node_new
    
    # Actualizamos la tabla de codificación añadiendo a la izquierda de los códigos
    # correspondientes un 0 o un 1 según si quedan a la derecha o a la izquierda
    # en el nuevo nodo del árbol
    for s in first[1]:
        table[s] = '0' + table[s]
    for s in second[1]:
        table[s] = '1' + table[s]
    
    # Introducimos el nuevo elemento en la lista con la probabilidad correspondiente
    distr.put((probab_new, state_new))
    
    return distr 

# Construye un árbol de Huffman y su tabla de codificación a partir de la lista
# de caracteres y su probabilidad de aparición
def huffman_tree(distr):
    treeList = {}
    table = {}
    for s in distr.queue:
        treeList[s[1]] = TreeNode(s[1],None,None)
        table[s[1]] = ''
    while len(distr.queue) > 1:
        distr = huffman_branch(distr, treeList, table)
    return list(treeList.values())[0], table

# Codifica una palabra utilizando la codificación de Huffman dada por parámetro
def code(word, code):
    res = ''
    for c in word:
        res += code[c]
    return res

# Decodifica una palabra utilizando el árbol de Huffman dado por parámetro
def decode(word, tree):
    res = ''
    node = tree
    for d in word:
        if d == '0':
            node = node.left
        else:
            node = node.right
        if len(node.root) == 1:
            res += node.root
            node = tree
    return res

# Leemos los ficheros con el texto en el que basaremos la codificación
with open('auxiliar_en_pract2.txt', 'r') as file:
      en = file.read()
with open('auxiliar_es_pract2.txt', 'r') as file:
      es = file.read()

# Pasamos todas las letras a minúsculas
en = en.lower()
es = es.lower()

# Contamos cuantas letras hay en cada texto
tab_en = Counter(en)
tab_es = Counter(es)

# Construimos una cola de prioridad con los caracteres y su frecuencia. De esta
# forma, se mantendrá ordenada aunque introduzcamos elementos nuevos y devolverá
# en primer lugar el que tenga la menor probabilidad
total_en = np.sum(list(tab_en.values()))
distr_en = PriorityQueue()
for c in tab_en:
    distr_en.put((tab_en[c]/float(total_en), c))

total_es = np.sum(list(tab_es.values()))
distr_es = PriorityQueue()
for c in tab_es:
    distr_es.put((tab_es[c]/float(total_es), c))
 
# Llamamos a huffman_tree para construir el árbol de Huffman y la tabla de
# codificación para cada idioma.
tree_en, table_en = huffman_tree(distr_en)
tree_es, table_es = huffman_tree(distr_es)
print("El árbol de Huffman en inglés es:")
show_tree(tree_en)
print("La tabla de codificación de Huffman en inglés es:")
print(table_en)
print("El árbol de Huffman en castellano es:")
show_tree(tree_es)
print("La tabla de codificación de Huffman en castellano es:")
print(table_es)

# Calculamos la longitud media de cada codificación y  su entropía y comprobamos
# que se cumple el primer teorema de Shannon
lonH_en = np.sum([tab_en[a] * len(table_en[a]) for a in tab_en.keys()]) / float(total_en)
ent_en = -1 * np.sum([(a / float(total_en)) * np.log2(a / float(total_en)) for a in tab_en.values()])
lonH_es = np.sum([tab_es[a] * len(table_es[a]) for a in tab_es.keys()]) / float(total_es)
ent_es = -1 * np.sum([(a / float(total_es)) * np.log2(a / float(total_es)) for a in tab_es.values()])
print("La longitud media usando la codificación de Huffman en inglés es:", lonH_en)
print("La entropía en inglés es:", ent_en)
print("En efecto, se cumple el primer teorema de Shannon:", ent_en, "<", lonH_en, "<", ent_en + 1)
print("La longitud media usando la codificación de Huffman en castellano es:", lonH_es)
print("La entropía en castellano es:", ent_es)
print("En efecto, se cumple el primer teorema de Shannon:", ent_es, "<", lonH_es, "<", ent_es + 1)

# Codificamos la palabra 'fractal' en ambos idiomas y comparamos su longitud con
# la que se obtendría con la codificación binaria trivial
cod_en = code('fractal', table_en)
cod_es = code('fractal', table_es)
print("Con la codificación en inglés, la palabra 'fractal' se codifica como:", cod_en)
print("Su longitud es", str(len(cod_en)) + ", mientras que con la codificación trivial sería",
      np.ceil(np.log2(total_en)) * len('fractal'))
print("Con la codificación en castellano, la palabra 'fractal' se codifica como:", cod_es)
print("Su longitud es", str(len(cod_es)) + ", mientras que con la codificación trivial sería",
      np.ceil(np.log2(total_es)) * len('fractal'))

# Decodificamos la palabra 1010100001111011111100 haciendo uso del árbol de
# Huffman obtenido a partir del texto en inglés
word = decode('1010100001111011111100', tree_en)
word_en = decode(cod_en, tree_en)
word_es = decode(cod_es, tree_es)
print("La palabra correspondiente al código '1010100001111011111100' utilizando la codificación"
      + " en inglés es", word)
print("Al decodificar el código '" + cod_en + "' utilizando la codificación en inglés"
      + " obtenemos nuevamente", "'" + word_en + "'")
print("Al decodificar el código '" + cod_es + "' utilizando la codificación en castellano"
      + " obtenemos nuevamente", "'" + word_es + "'")


# Calculamos la probabilidad de aparición de cada letra y la probabilidad  acumulada
prob_en = tab_en.copy()
for a in prob_en.keys():
    prob_en[a] /= float(total_en)
prob_en = sorted(prob_en.items(), key=lambda kv : kv[1])
acum_en = [prob_en[0][1]]
for i in range(1,len(prob_en)):
    acum_en.append(acum_en[i-1] + prob_en[i][1])
    
# Representamos la curva de Lorenz
plt.figure(figsize=(10,10))
plt.plot(np.linspace(0, 1, len(acum_en)+1), [0] + acum_en, 'r', label='Curva de Lorenz $y(x)$')
plt.plot(np.linspace(0, 1, len(acum_en)+1), np.linspace(0, 1, len(acum_en)+1), 'g', label='$y = x$')
plt.title("Función de Lorenz para la variable $S_{english}$")
plt.legend(loc="lower right")
plt.savefig("Lorenz.png")
plt.show()

# Calculamos el índice de Gini y la diversidad 2D de Hill para la variable aleatoria
gi = 1 - 2 / len(acum_en) * np.sum(acum_en)
dHill = 1 / np.sum(np.array([a[1] for a in prob_en])**2)
print("El índice de Gini es GI =", gi)
print("La diversidad 2D de Hill es 2D =", dHill)