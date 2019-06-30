#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 14:38:37 2019

@author: king
"""
#importing the libraries
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

# Importing the libraries of openai gym and doom
import gym 
from gym.wrappers import SkipWrapper
from ppaquette_gym_doom.wrappers.action_space import ToDiscrete

#Fetching important files
import image_preprocessing , experience_replay

#Bulding the AI
#Making the brain of AI
class CNN(nn.Module):
    
    def __init__(self , number_actions):
        self.convolution1 = nn.Conv2d(in_channels = 1 , out_channels = 32 , kernel_size = 5)
        self.convolution2 = nn.Conv2d(in_channels = 32 , out_channels = 32 , kernel_size = 3)
        self.convolution3 = nn.Conv2d(in_channels = 32 , out_channels = 64 , kernel_size = 2)
        self.fc1 = nn.Linear(in_features = neurons_count(1,80,80) , out_features = 40)
        self.fc2 = nn.Linear(in_features = 40 , out_features = number_actions)  
        
    def neurons_count(self , image_size):
        x = Variable(torch.rand(1, *image_size))
        x = F.relu(F.max_pool2d(self.convolution1(x) , 3, 2))
        x = F.relu(F.max_pool2d(self.convolution2(x) , 3, 2))
        x = F.relu(F.max_pool2d(self.convolution3(x) , 3, 2))
        return x.data.view(1,-1).size(1)
    
    def forward(self , x):
        x = F.relu(F.max_pool2d(self.convolution1(x) , 3, 2))
        x = F.relu(F.max_pool2d(self.convolution2(x) , 3, 2))
        x = F.relu(F.max_pool2d(self.convolution3(x) , 3, 2))
        x = x.view(x.size(0),-1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
#Making body of AI
class SoftmaxBody(nn.Module):
    
    def __init(self , T):
        super(SoftmaxBody , self).__init__()
        self.T = T
        
    def forward(self , brain_output):
        probs  = F.softmax(brain_output * self.T)
        actions = probs.multinomial()
        return actions
    
# Making AI
class AI:
    
    def __init(self , brain , body):
        self.brain = brain
        self.body = body
        
    def __call__(self , inputs):
        input = Variable(torch.from_numpy(np.array(inputs,dtype = np.float32)))
        brain_output = self.brain(input)
        actions = self.body(brain_output)
        return actions.data.numpy()
    

#Training the AI with deep convolution Q learning

#Creating environment for doom
doom_env = image_preprocessing.PreprocessImage(SkipWrapper(4)(ToDiscrete("minimal")(gym.make("ppaquette/DoomCorridor-v0"))), width = 80 , height = 80 , grayscale = True)
doom_env = gym.wrappers.Monitor(doom_env , "video" , force = True)
number_actions = doom_env.action_space.n

#Bulding the AI
cnn = CNN(number_actions)
softmax_body = SoftmaxBody(T = 1.0)
ai = AI(brain = cnn , body = softmax_body)

#Setting up experience replay
n_steps = experience_replay.NStepProgress(env = doom_env , ai = ai , n_step = 10)
memory = experience_replay.ReplayMemory(n_steps = n_steps , capacity = 10000)

#Implementing Eligibility trace 
def eligibility_trace(batch):
    gamma = 0.99
    inputs = []
    targets = []
    for series in batch:
        input = Variable(torch.from_numpy(np.array([series[0].state , series[-1].state],dtype = np.float32)))
        outputs = CNN(input)
        cum_reward1 = 0.0 if series[-1].done else outputs[1].data.max()
        for step in reversed(series[:-1]):
            cum_reward1 = step.reward + gamma*cum_reward1
        state = series[0].state
        target = outputs[0].data
        target[series[0].action] = cum_reward1
        inputs.append(state)
        targets.append(target)
        return torch.from_numpy(np.array(inputs , dtype = np.float32)) , torch.stack(targets)
  
#Making moving average on 100 steps
class MA:
    
    def __init__(self , size):
        self.list_of_rewards = []
        self.size = size
    
    def add (self , rewards):
        if isinstance(rewards , list):
            self.list_of_rewards += rewards
        else:       
            self.list_of_rewards.append(rewards)
        while len(self.list_of_rewards) > self.size:
            del self.list_of_rewards[0]
    
    def average(self):
        return np.mean(self.list_of_rewards)
ma = MA(100)

#Training the AI
loss =  nn.MSELoss()
optimzer = optim.Adam(cnn.parameters() , lr = 0.001)
nb_epoches = 100
for epoch in range (1, nb_epoches + 1):
    memory.run_steps(200)
    for batch in memory.sample_batch(128):
        inputs , targets = eligibility_trace(batch)
        inputs , targets = Variable(inputs) , Variable(targets)
        predictions = CNN(inputs)
        loss_value = loss(predictions , targets)
        optimzer.zero_grad()
        loss_value.backward()
        optimzer.step()
    reward_step = n_steps.rewards_steps()
    ma.add(reward_step)
    average_reward = ma.average()
    print("Average Reward : %s , Number of epoches : %s " % (str(average_reward) , str(epoch)))
    if average_reward > 1500:
        print("Congratulations : you have win the game")
        break
    
# closing the Doom Enviourment
doom_env.close()

    

















