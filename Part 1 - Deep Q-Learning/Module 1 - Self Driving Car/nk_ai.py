#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 08:18:56 2019

@author: king
"""
#Importing usefull libraries

import numpy as np
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable

#Creation of Neural Network
class Network(nn.Module):
    
    def __init__(self,input_size ,nb_action):
        super(Network,self).__init__()
        self.input_size = input_size
        self.nb_action = nb_action
        self.fc1 = nn.Linear(input_size,30)
        self.fc2 = nn.Linear(30,nb_action)
        
    def forward(self , state): # activate neural network
        x = F.relu(self.fc1(state)) #relu (reactifier function )use for activate hidden neurons layer
        q_values = self.fc2(x)
        return q_values
    
#Implementing Experience replay 
class ReplayMemory(object):
    
    def __init__(self , capacity):
        self.capacity = capacity
        self.memory = []
    
    def push(self , event):
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]
        
    def sample(self , batch_size):
        #list((1,2,3,),(4,5,6)) then zip(*list((1,4),(2,5),(3,6)))
        samples = zip(*random.sample(self.memory,batch_size))
        return map(lambda x : Variable(torch.cat(x,0)),samples)

#implementing deep q learning class
class Dqn():
    
    def __init__(self , input_size , nb_action , gamma):
        self.gamma = gamma
        self.reward_window = []
        self.memory = ReplayMemory(100000)
        self.model = Network(input_size , nb_action)
        self.optimizer = optim.Adam(self.model.parameters(),lr = 0.001)
        self.last_state = torch.Tensor(input_size).unsqueeze(0)
        self.last_action = 0 
        self.last_reward = 0
    
    def select_action(self,state):
        probs = F.softmax(self.model(Variable(state , volatile = True))*100) # T = 7 (Temprature) T=0 if we don't want brain
        action = probs.multinomial()
        return action.data[0,0]
    
    def learn(self , batch_state , batch_next_state , batch_action , batch_reward):
        output = self.model(batch_state).gether(1,batch_action.unsqueeze(1)).squeeze(1)
        next_output = self.model(batch_next_state).detach().max(1)[0]
        target = batch_reward + self.gamma*next_output
        td_loss = F.smooth_l1_loss(output , target)
        self.optimizer.zero_grad()
        td_loss.backward(retain_variables = True)
        self.optimizer.step()
        
    def update(self , reward , new_signal):
        new_state = torch.Tensor(new_signal).float().unsqueeze(0)
        self.memory.push(self.last_state , new_state , torch.LongTensor([int(self.last_action)]),torch.Tensor([self.last_reward]))
        action = self.select_action(new_state)
        if len(self.memory.memory) > 100:
            batch_state , batch_next_state , batch_action , batch_reward = self.memory.sample(100)
            self.learn(batch_state , batch_next_state , batch_action , batch_reward)
        self.last_state = new_state
        self.last_action = action
        self.last_reward = reward
        self.reward_window.append(reward)
        if len(self.reward_window) > 1000:
            del self.reward_window[0]
        return action
    
    def score(self):
        return sum(self.reward_window)/(len(self.reward_window)+1)
    
    def save(self):
        torch.save({'state_dict' : self.model.state_dict(),
                    'optimizer' : self.model.state_dict,
                    },'last_brain.pth')
    
    def load(self):
        if os.path.isfile('last_brain.pth'):
            print("=> Loading checkpoints ...")
            checkpoint = torch.load('last_brain.pth')
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("done !")
        else :
            print("=> No checkpoint found")
            
    
            
    
    
    
        
            
        
    
    
            























