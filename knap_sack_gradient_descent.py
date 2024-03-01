# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 19:46:07 2024

@author: scott
"""

from bitarray import bitarray 
import math


def sigmoid(x):
    try:
        return 1/(1+math.exp(-x))
    except:
        raise Exception("number: "+str(x))

def fmap_vec(x, f):
    return [f(i) for i in x]

def dot(x,y):
    return sum(i*j for i,j in zip(x,y))

def loss(vec, weights,values, cap, l1,l2, l3):
    sig = fmap_vec(vec, sigmoid)
    total_weight = dot(sig, weights)
    total_values = dot(sig, values)
    part_1 = l1*(total_weight-cap)*(total_weight-cap)
    part_2 = -l2*total_values*total_values
    part_3 = l3*sum((sig[i]-1)**2 for i in range(len(weights)))
    return part_1+part_2+part_3

def value_score(vec, values):
    sig = fmap_vec(vec,sigmoid)
    return dot(sig,values)

def knap_sack_grad(weights, values, cap, l1, l2, l3,learning_rate,iterations=1000):
    k = len(weights)
    vec = [0 for _ in range(k)]
    print(loss(vec, weights,values, cap, l1,l2, l3))
    for i in range(iterations):
        sigmoid_vec = fmap_vec(vec, sigmoid)
        total_weight = dot(sigmoid_vec, weights)
        total_value = dot(sigmoid_vec, values)
        for j in range(k):
            p = sigmoid_vec[j]
            part_1 = p*(1-p)*weights[j]*(total_weight-cap)
            part_2 = p*(1-p)*values[j]*total_value
            part_3 = p*(1-p)*(p-1)
            vec[j] = vec[j]-learning_rate*(l1*part_1-l2*part_2+l3*part_3)
    print(loss(vec, weights,values, cap, l1,l2, l3))
    return vec