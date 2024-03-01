# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 13:07:40 2024

@author: scott
"""

import random 
from abc import ABC,abstractmethod 
from bisect import bisect_left,insort
import bitarray
import statistics as stats

class Genetic:
    
    def generate_gene():
        pass
    
    @abstractmethod 
    def fitness(self):
        pass
    
    @abstractmethod
    def combine(self, other):
        pass
    
    @abstractmethod
    def mutate(self):
        pass
    
    
class one_zero_knap_sack(Genetic):
    def __init__(self, weights, values, capacity, binary):
        self.weights = weights
        self.values = values 
        self.capacity = capacity
        self.bit_array = binary
        self.size = len(self.values)
        self.fit = None
        self.mutation_rate = 1

    def fitness(self):
        if self.fit is not None:
            return self.fit 
        weight = 0
        v = 0 
        for i,b in enumerate(self.bit_array):
            if b == 1:
                weight += self.weights[i]
                v += self.values[i]
        if weight > self.capacity:
            self.fit = 0 
            return self.fit 
        self.fit = v
        return v

    def combine(self, other):
        weights = self.weights 
        values = self.values 
        capacity = self.capacity
        bit_array = bitarray.bitarray(len(self.values))
        for i in range(len(self.bit_array)):
            if self.bit_array[i] == other.bit_array[i]:
                bit_array[i] = self.bit_array[i] 
            else:
                bit_array[i] = random.randint(0,1)
        g = one_zero_knap_sack(weights,values, capacity,bit_array)
        if random.randint(0,1):
            g.mutation_rate = self.mutation_rate
        else:
            g.mutation_rate = other.mutation_rate
        return g

    
    def copy(self):
        weights = self.weights
        values = self.values 
        capacity = self.capacity
        bit_array = bitarray.bitarray(len(self.values)) 
        for i,b in enumerate(self.bit_array):
            bit_array[i] = b
        g = one_zero_knap_sack(weights,values, capacity,bit_array )
        g.mutation_rate = self.mutation_rate
        return g
    
    def mutate(self):
        g = self.copy()
        for i in range(self.mutation_rate):
            index = random.randint(0,g.size-1)
            if g.bit_array[index] == 0:
                g.bit_array[index] = 1
            else:
                g.bit_array[index] = 0
        
        if random.randint(0,10) == 0:
            if random.randint(0,1) == 0:
                g.mutation_rate += 1 
            else:
                g.mutation_rate -= 1
                g.mutation_rate = max(0, g.mutation_rate)
        return g
    
class Selector:
    
    @abstractmethod 
    def select(self, pool):
        pass 
    
    
class CircleSelector(Selector):
    def __init__(self, elite_num, kill_num, max_random):
        self.elite_num = elite_num
        self.kill_num = kill_num 
        self.max_random = max_random
        
    def construct_circle(self, pool):
        total_fitness = sum(p.fitness() for p in pool)
        circle = []
        s = 0
        if total_fitness == 0:
            return [(i+1)/len(pool) for i in range(len(pool))]
        for p in pool:
            s += p.fitness()/total_fitness 
            circle.append(s)
        return circle 

    def select(self, pool):
        circle = self.construct_circle(pool)
        i = self.elite_num
        j = len(pool)-1-self.kill_num
        # pick random number 
        num = random.randint(0, self.max_random)/self.max_random 
        first_index = bisect_left(circle, num, i, j)
        num = random.randint(0, self.max_random)/self.max_random 
        second_index = bisect_left(circle, num, i, j)
        return pool[first_index], pool[second_index]
        
    
def evolve(inital_genetics, iterations=1000):
    lst = sorted(inital_genetics,key=lambda x : -x.fitness())
    selector = CircleSelector(0, 2, 1<<30)
    best_fitness = 0
    points = []
    for i in range(iterations):
        s1,s2 = selector.select(lst)
        s3 = s1.combine(s2)
        s3 = s3.mutate() 
        lst.pop()
        insort(lst, s3, key=lambda x : -x.fitness())
        best_fitness = lst[0].fitness() 
        mean_mutation = stats.mean(l.mutation_rate for l in lst)
        mean_fitness = stats.mean(l.fitness() for l in lst)
        points.append((i, best_fitness,mean_fitness , mean_mutation))
        
    return lst,points


def random_bits(n):
    b = bitarray.bitarray(n)
    for i in range(n):
        b[i] = random.randint(0,1)
    return b 


def sample_from_distribution(vector, weights,values, cap, n):
    genes = []
    for i in range(n):
        b = bitarray.bitarray(len(weights))
        for i,v in enumerate(vector):
            if random.randint(1,1<<30) <= int((1<<30)*v):
                b[i] = 1
        g = one_zero_knap_sack(weights, values, cap, b)
        g.mutation_rate = random.randint(1,100)
        genes.append(g)
    return genes

def generate_random_animals(weights, values, cap, n):
    genes = []
    for i in range(n):
        bits = bitarray.bitarray(len(weights))
        g = one_zero_knap_sack(weights, values, cap, bits)
        g.mutation_rate = random.randint(1,100)
        for i in range(random.randint(0, len(weights))):
            g.bit_array[random.randint(0,len(weights)-1)] = 1
        genes.append(g) 
    genes = sorted(genes,key=lambda x : -x.fitness())
    return genes

def generate_random_animals_50(weights, values, cap, n):
    genes = []
    for i in range(n):
        bits = bitarray.bitarray(len(weights))
        g = one_zero_knap_sack(weights, values, cap, bits)
        g.mutation_rate = random.randint(1,100)
        for i in range(10):
            g.bit_array[random.randint(0,len(weights)-1)] = 1
        genes.append(g) 
    genes = sorted(genes,key=lambda x : -x.fitness())
    return genes

def generate_random_animals_zeros(weights, values, cap, n):
    genes = []
    for i in range(n):
        bits = bitarray.bitarray(len(weights))
        g = one_zero_knap_sack(weights, values, cap, bits)
        g.mutation_rate = random.randint(1,100)
        genes.append(g) 
    genes = sorted(genes,key=lambda x : -x.fitness())
    return genes

def random_evolve(weights, values, cap, iterations=1000):
    best = None
    best_val = 0
    n = len(values)
    for i in range(iterations):
        bits = random_bits(n)
        g = one_zero_knap_sack(weights, values, cap, bits)
        
        if best is None or best_val < g.fitness():
            best_val = g.fitness()
            best = g 
            
    return best

#https://www.geeksforgeeks.org/0-1-knapsack-problem-dp-10/
def knapsack(wt, val, W, n, t=None): 
    if t is None:
        t = [[-1 for _ in range(W+1)] for _ in range(n+1)]
    # base conditions 
    if n == 0 or W == 0: 
        return 0
    if t[n][W] != -1: 
        return t[n][W] 
  
    # choice diagram code 
    if wt[n-1] <= W: 
        t[n][W] = max( 
            val[n-1] + knapsack( 
                wt, val, W-wt[n-1], n-1, t), 
            knapsack(wt, val, W, n-1, t)) 
        return t[n][W] 
    elif wt[n-1] > W: 
        t[n][W] = knapsack(wt, val, W, n-1, t) 
        return t[n][W]



