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
    P(c_1=1,c_2=1,c_3=1) = 0
    P(c_1=1,c_2=1,c_3=0) = P(c_1)P(c_2)
   
    So P(c_1), P(c_2) are sources.
    P(c_3|c_2,c_1) = 0 if c_2 
    P(c_4|c_1,c_2,c_3)
'''
import numpy as np
import pandas as pd
    


class Prob:
    def __init__(self, params):
        self.params = list(params)
        self.alpha = 0.1
        
        def normalize_fn(self): #designed to work with Prob_array too :) That's why we count backwards.
            self.normalization_factor = self.count
            for i in range(len(self.params)):
                self.normalization_factor = self.normalization_factor.sum(axis = -1-i, keepdims = True)
        self.normalize = normalize_fn
        
        self.count = np.zeros(params)
    
    def update(self, data):
        '''

        Parameters
        ----------
        data : array [params]
            [p1,p2, ...]

        '''
        if hasattr(self,'count'):
            self.count = self.alpha * data + self.count * (1 - self.alpha)
        else:
            self.count = data
        self.normalize()    
        
    def __call__(self, params = None):
        
        return self.count / self.normalization_factor


class Prob_array(Prob):
    def __init__(self, params, shape):
        super().__init__(params)
        self.shape = list(shape)
        self.count = np.zeros(shape + params)
        
    def append(self,shape):
        self.count = np.vstack(self.count, np.zeros(shape + self.params))
        
    def pop(self, i): #don't worry about return value. Don't use it.
        self.count = np.delete(self.count, i, 0)

class Cond_Prob_array(Prob_array):
    def __init__(self, params, cond_params, shape):
        super().__init__(params,shape)
        self.cond_params = list(cond_params)
        self.count = np.zeros(shape + params+cond_params)
    def append(self,shape):
        self.count = np.vstack(self.count, np.zeros(shape + self.params+self.cond_params))

    
    
class super_list: #basically a wrapper around pd.Series making it act more like a list
    def __init__(self,*args):
        self.data = pd.Series(index = args, data = np.arange(len(args)))
        
        self.__contains__ = self.data.__contains__ #makes 'in' work
    
    def append(self, value):
        self.data.loc[value] = len(self.data)
    
    
    def __getitem__(self,idx):
        return self.data[idx].values

class sparse_array: #special case of array
    def __init__(self):
        
        return 
    def __getitem__(self,*args):
        

class BayesianModel:

    def __init__(self, num_keys, line_size):
        self.num_keys = num_keys
        self.line_size = line_size
        self.car_ids = super_list()
        self.threshold = 0.9
        
        self.P_C = Cond_Prob_array([2],[line_size],[0])
        self.P_S = Prob_array([2],[num_keys])
        self.P_L = Prob([line_size])
        self.P_K = Cond_Prob_array([2],[2],[num_keys])
        self.P_Y = Cond_Prob_array([2]*num_keys,[2]*(num_keys+1),[0]) #[j,c,k0..kn,y0,..yn]
        
        
    def process_run(self, run):
        
        #initialize
        for c in run['cars_seen']:
            if c not in self.car_ids:
                self.P_C.append()
                self.P_Y.append()
                    
                self.car_ids.append(c)
        
        
        
        #Translate and estimate    Have S and C.  Need K and Y.
        
        
        data = dict()
        S = np.isin(np.arange(len(self.P_S)),run['keys_sensed'],assume_unique=True).astype(int)
        data['S'] = np.vstack([S,1-S]).T
        
        data['S,K'] = np.array([(1-S)*(self.P_K(0)),S * 0,(1-S) * self.P_K(0), S ]).reshape(2,2,10).T
        
        l = len(run['cars_seen'])
        data['L'] = np.zeros(self.line_size)
        data['L'][l] = 1
        
        c_1 = self.car_ids[run['cars_seen']]
        c_0 = np.setdiff1d(np.arange(len(self.car_ids)),c_1)
        
        data['L,C'] = np.zeros([len(self.P_C),self.line_size,2])
        data['L,C'][c_0,l,0] = 1
        data['L,C'][c_1,l,1] = 1
        
        
        K = data['S,K'].sum(1) #[i,k]
        C = data['L,C'].sum(1) #[j,c]
        K_tile = np.tile(K[i], [2]*(self.num_keys-1)+[1])
        for i in range(self.num_keys - 1):
            K_tile = K_tile * np.swapaxes(np.tile(K[i], [2]*(self.num_keys-1)+[1]),-1,-i)
        #[j,c,k0..kn,y0,..yn]
        data['K,C,Y'] =  np.tile(np.expand_dims(np.expand_dims(K_tile,-1),-1),[2]*self.num_keys + [1]*self.num_keys + [2,len(self.P_C)]).T
        data['K,C,Y'] *= np.tile(C.T, [2]*self.num_keys * 2 + [1,1]).T
        data['K,C,Y'] *= self.P_Y()
        

        
        #update 
        self.P_L.update(data['L'])
        
        self.P_S.update(data['S'])
        
        self.P_C.update(data['L,C'])
        
        self.P_K.update(data['S,K'])
        
        self.P_Y.update(data['K,C,Y'])
        
              
        #make decision
        #  p(x_j|D) = p(y_0j,..,y_nj|D)
        i = [slice(len(data['K,C,Y'])), slice(len(self.P_C)) ]
        for _ in range(self.num_keys):
            i.append(slice(2))
        for _ in range(self.num_keys):
            i.append(0)
        x = data['K,C,Y'][i].sum(1)
        for _ in range(self.num_keys):
            x = x.sum(1)
            
        
        return np.where(r > self.threshold)[0]
        
            

