#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 18:29:53 2023

@author: maya cutkosky
"""
from collections.abc import Iterable
import numpy as np


def pick(p):
    if isinstance(p, Iterable):
        return np.where(np.random.rand(len(p)) < p)[0]
    else:
        return np.random.rand() < p

def rand(start,end,n):
    return np.random.rand(n)*(end-start) + start
    
class Environment:
    '''
    Car object:
        P(exists|t) 
        P(contains_key|t)
        P(dies|age)
    
    Person object:
        preferences: dict
        cars
    '''
    
    class Car:
        def __init__(self, owner):
            self.owner = owner
            self.alpha = self.owner.preferences['alpha'] * (np.random.randn()/10+1)
            self.age = self.owner.preferences['age'] * (np.random.randn()/10+1)
            self.P_exists = owner.preferences['drive']
            self.P_contains_key = owner.preferences['carpool']
            
        def __getitem__(self,item,t):
            item = item.rpartition('|')[0].replace('(','_')
            return getattr(self,item)[t]
        
        def P_dies(self):
            return 1 - np.exp(-self.age /self.alpha )
        
    class Person:
        def __init__(self, config, ptype):
            self.type = ptype
            self.preferences = dict()
            self.preferences['alpha'] = np.random.normal(*config[ptype]['alpha'])
            self.preferences['age'] = np.random.normal(*config[ptype]['age'])
            self.preferences['drive'] = np.clip(np.random.normal(config[ptype]['drive'][0], config[ptype]['drive'][1],[config['cycle_size'],config['num_keys']]),0,1) 
            
            if ptype == 'person':
                r = np.random.rand(config['cycle_size'], config['num_keys'])
                self.preferences['carpool'] = np.clip((r-0.5+np.random.rand(config['num_keys'])),0,1)
            elif ptype == 'friend':
                r = np.random.rand([config['cycle_size'], config['num_keys']])
                self.preferences['carpool'] = np.clip((r-0.8+np.random.rand(config['num_keys'])),0,1)
            elif ptype == 'malicious':
                self.preferences['carpool'] = np.zeros([config['cycle_size'],config['num_keys']])
        
    
    def __init__(self,config):
        self.people = []
        #Create people with keys
        for i in range(config['num_keys']):
            self.people.append(self.Person(config,'person'))
        
        #Create friends of people with keys
        for i in range(config['num_friends']):
            self.people.append(self.Person(config, 'friend'))
        
        #Create malicious people without keys
        for i in range(config['num_malicious']):
            self.people.append(self.Person(config,'malicious'))
        
        #Make car list
        self.cars = []
        for p in self.people:
            for i in range(np.random.choice(config['num_cars_vals'], p = config['num_cars_p'])):
                self.cars.append(self.Car(p))
            
        
        #other parameters
        self.line = []
        self.P_sense = config['P(sense)']
        self.config = config
        self.cycle_size = config['cycle_size']
    
    def run(self,t):
        t = t%self.cycle_size
        output = dict()
        #add cars to line
        for car in self.cars:
            if pick(car.P_exists[t]): 
                self.line.append(car)
        
        #look at cars going through
        output['cars_seen'] = []
        output['keys_exist'] = []
        output['keys_sensed'] = 0
        for i in range(self.k):
            if self.line:
                car = self.line.pop(0)
                output['cars_seen'].append(id(car))
                keys = pick(car.P_contains_key[t])
                output['keys_exist'].append(bool(len(keys)))
                output['keys_sensed'] += len(keys)
        
        
        #update cars
        for car in self.cars:
            if pick(car['P(dies|age)']):
                self.cars.remove(car)
                #high chance of buying new car if old one dies
                self.cars.append(self.Car(car.owner))
            else:
                car.age += 1
            
        
        return output
        



def score_guess(run_output, guess):
    '''
    

    Parameters
    ----------
    run_output : dict
        Output of Environment.run
    guess : vector
        Guess of which cars are passing without key. Boolean or int.
    

    Returns
    -------
    score : dict
        Describes how good our guess was

    '''
    
    truth = run_output['keys_exist']
    score =dict()
    score['missed'] = len([1 for i,j in zip(truth,guess) if i == 1 and j == 0]) / len(guess)
    score['wrong guess'] = len([1 for i,j in zip(truth,guess) if i == 1 and j == 0]) / len(guess)
    
    return score






    
