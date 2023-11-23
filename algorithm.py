#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 19:00:27 2023

@author: maya cutkosky
"""


'''
Problem information:
    At each time t, there exists: [c1,c2,c3,c4], [k1,k2,k3,k4], where k_i \in {0,1} and c_i \neq c_j
    We want 
    
    Let k = 1,
    States are: [(c1,0)],[(c1,1)], [(c2,0)], [(c2, 1)] \ldots
    Want P()

State: [(c1,k1,y1),(c2,k2,y2)]

c and o are known.
Want P(k |c, y) 

P(k_i | c,y) = 0 if c_i = 0
P(k_i | c,y) = 1 if y = sum(c) and c_i =  1
   Else, dependent on prior belief.

P( k_i | c ,y) = P( c,y | k_i)P(k_i) / P(c,y)

P(y = 0 | sum(c)=0) = 1
P(c_i = 0) = ?
P(y = 1 to k  | c_i = 1) = ?

How are c_i, k_i connected to c_j,k_j

Use (false) assumption:
    c_i and c_j only dependent based on k.
    sum(c) = k.
    Can make network c_1 -> c_2 \ldots
'''


class Prob:
    def __init__(self, *prob_params):
        self.data = 0
        self.total = 0
        self.prob = prob_params
    
    def add(x):
        self.data = self.data *self.total + x
        self.total += 1
        self.data /= self.total
        

class BeyesianModeling:

    def __init__(self):
        self.initializer = [2,2]
        self.P = dict() 
    
    def process_run(self, run):
        
        #initialize
        for c in run['car_seen']:
            if c not in self.P.keys():
                self.P[c] = [self.initialiizer]*4
            
        #Update
        P(c=0|D) = P(D|c=1) * P(c=1)/ sum( P(D|c)*P(c))
        4 cars, c = 0,5
        
        #make decision
        for c in run['car_seen']
            #calculate P(k|c,y) = P(k,c,y) / P(c,y) = P(c)P(k|c)*P(y|k)/ sum_k P(c)P(k|c)*P(y|k)
            
            


R = 
T = 
U = 
lam = 
    

    def bellman_update(self,R,T,U, lam): #R is independent of action, so can be taken outside the max equation.
        A = lam * np.sum(T * U,axis = -1)
        return R + np.max(A, axis=0), np.argmax(A, axis=0)


    def value_update(self):
        U = np.zeros([11,3])
        policy = np.empty([10,3])
        for i in range(10):
            U[i+1], policy[i] = bellman_update(R,T,U[i], lam)