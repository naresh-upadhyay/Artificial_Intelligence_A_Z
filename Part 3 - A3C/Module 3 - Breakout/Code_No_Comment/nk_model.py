#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 01:21:28 2019

@author: king
"""
# AI for breakout game

#Importing the libraries
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Initialize and setting up the varience of tensor of weights
def normalized_column_initializer(weights , std = 1.0):
    out = torch.randn(weights.size())
    out *= std / torch.sqrt(out.pow(2).sum(1).expand_as(out))
    return out

# Initalize the weight of neural network for an optimal learning
def weight_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4])*weight_shape[0]
        w_bound = np.sqrt(6. / fan_in + fan_out)
        m.weight.data.uniform_(-w_bound , w_bound)
        m.bias.data.fill_(0)
    elif classname.find("Linear") != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / fan_in + fan_out)
        m.weight.data.uniform_(-w_bound , w_bound)
        m.bias.data.fill_(0)
        
# Making the A3C brain
class ActorCritic(torch.nn.Module):
    
    def __init__(self , num_input , action_space):
        super(ActorCritic , self).__init__()
        self.convolution1 = nn.Conv2d(num_input , 32 , 3, stride = 2 , padding = 1)
        self.convolution2 = nn.Conv2d(32 , 32 , 3, stride = 2 , padding = 1)
        self.convolution3 = nn.Conv2d(32 , 32 , 3, stride = 2 , padding = 1)
        self.convolution4 = nn.Conv2d(32 , 32 , 3, stride = 2 , padding = 1)
        self.lstm = nn.LSTMCell(32*3*3 , 256)
        num_ouputs = action_space.n
        self.actor_linear = nn.Linear(256 , num_ouputs)
        self.critic_linear = nn.Linear(256 , 1)
        self.apply(weight_init)
        self.actor_linear.weight.data = normalized_column_initializer(self.actor_linear.weight.data , 0.01)
        self.actor_linear.bias.data.fill_(0)
        self.critic_linear.weight.data = normalized_column_initializer(self.critic_linear.weight.data , 1.0)
        self.critic_linear.bias.data.fill_(0)
        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)        
        self.train()
        
        
    def forward(self , inputs):
        inputs , (hx , cx) = inputs
        x = F.elu(self.convolution1(inputs))
        x = F.elu(self.convolution2(x))
        x = F.elu(self.convolution3(x))
        x = F.elu(self.convolution4(x))
        x = x.view(-1 , 32*3*3)
        (hx , cx ) = self.lstm(x , (hx , cx))
        x = hx
        return self.critic_linear(x) , self.actor_linear(x) , (hx , cx)
    





















