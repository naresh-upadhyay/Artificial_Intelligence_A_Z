#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 02:33:03 2019

@author: king
"""

#Importing the libraries
import torch
import torch.nn.functional as F
from envs import create_atari_env
from model import ActorCritic
from torch.autograd import Variable

#Ensuring the shared module
def ensure_shared_model(model , shared_model):
    for param , shared_param in zip (model.parameters() , shared_model.parameters()):
        if shared_param.grad in None:
            return
        shared_param._grad = param.grad
        
#Train the model
def train(rank , params , shared_model , optimizer):
    torch.manual_seed(params.seed + rank)
    env = create_atari_env(params.env_name)
    env.seed(params.seed + rank)
    model = ActorCritic(env.observation_space.shape[0] , env.action_space)
    state = env.reset()
    state = torch.from_numpy(state)
    done = True
    episode_length = 0
    while True:
        episode_length += 1
        model.load_state_dict(shared_model.state_dict)
        if done:
            hx = Variable(torch.zeros(1 , 256))
            cx = Variable(torch.zeros(1 , 256))
        else:
            hx = Variable(hx.data)
            cx = Variable(cx.data)
        values = []
        log_probs = []
        rewards = []
        entropies = []
        for step in range(params.num_steps):
            value , action_value , (hx , cx) = model(Variable(state.unsqueeze[0]),(hx , cx))
            probs = F.softmax(action_value)
            log_prob = F.log_softmax(action_value)
            entropy = -(log_probs*probs).sum(1)
            entropies.append(entropy)
            action = probs.multinomial().data
            log_prob = log_prob.gather(1,Variable(action))
            values.append(value)
            log_probs.append(log_prob)
            state , reward , done = env.step(action.numpy())
            done = (done or episode_length >= params.max_episode_length)
            reward = max(min(reward , 1) , -1)
            if done:
                episode_length = 0
                state = env.reset()
            state = torch.from_numpy(state)
            rewards.append(reward)
            if done:
                break
        R = torch.zeros(1,1)
        if not done:
            value , _ , _ = model(Variable(state.unsqueeze[0]),(hx , cx))
            R = value.data
        values.append(Variable(R))
        policy_loss = 0
        value_loss = 0
        R = Variable(R)
        gae = torch.zeros(1,1)
        for i in reversed(range(len(rewards))):
            R = params.gamma * R + rewards[i]
            advantage = R - values[i]
            value_loss = value_loss + 0.5*advantage.pow(2)
            TD = rewards[i] + params.gamma * values[i+1].data - values[i].data
            gae = gae * params.gamma * params.tau + TD
            policy_loss = policy_loss - log_probs[i] * Variable(gae) - 0.01 * entropies[i]
            optimizer.zero_grad()
            (policy_loss + 0.5 * value_loss).backward()
            torch.nn.utils.clip_grad_norm(model.parameters() , 40)
            ensure_shared_model(model , shared_model)
            optimizer.step()
            
            
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    