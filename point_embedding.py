#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 11:13:13 2024

@author: ningnong
"""

from matplotlib import pyplot as plt
import numpy as np
from numpy import linalg as ln
import random 
from copy import deepcopy 

def list_to_matrix(edges, V): 
    matrix = [[None]*V for _ in range(V)]
    for u in range(V):
        matrix[u][u] = 0
    for u,v,w in edges:
        matrix[u][v] = w 
        matrix[v][u] = w
    return matrix

def loss(points, graph):
    V = len(graph)
    p = points
    s = 0
    for u in range(V):
        for v in range(V):
            part_1 = (np.array(p[u])-np.array(p[v]))
            n = ln.norm(part_1)
            s += (n**2-graph[u][v]**2)**2
    return s/(V*(V-1))

def embed(dimensions, graph,learning_rate=0.001,iterations=100):
    V = len(graph)
    points = [] 
    for u in range(V): 
        points.append([])
        for d in range(dimensions):
            points[u].append(random.randint(-10,10))
    
    for i in range(iterations): 
        points_c = deepcopy(points) 
        for p in range(V):
            w_diff = [None]*V
            for v in range(V): 
                weight_diff = sum((points_c[p][d]-points_c[v][d])**2 for d in range(dimensions))
                w_diff[v] = weight_diff-graph[p][v]**2
            
            for d in range(dimensions):
                s = 0 
                for v in range(V):
                    s += (points_c[p][d]-points_c[v][d])*w_diff[v]
                s = s/(V*(V-1))
                points[p][d] = points_c[p][d]-learning_rate*s
    return points
 



               
