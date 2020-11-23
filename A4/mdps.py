#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 15:19:13 2020

@author: jiajilu
"""

import mdptoolbox.example
import numpy as np
import gym
import copy


class ForestManagement:
    
    def __init__(self, state, epsilon, gamma):
        self.state = state
        self.epsilon = epsilon
        self.gamma = gamma
        self.P, self.R = mdptoolbox.example.forest(S=self.state, r1=500, r2=250)
        self.vi = mdptoolbox.mdp.ValueIteration(self.P, self.R, self.gamma, max_iter=10000)
        self.pi = mdptoolbox.mdp.PolicyIteration(self.P, self.R, self.gamma, max_iter=1000)
        self.ql = mdptoolbox.mdp.QLearning(self.P, self.R, self.gamma)

    def run(self):
        self.vi.setVerbose()
        self.vi.run()
        self.pi.setVerbose()
        self.pi.run()
        self.ql.setVerbose()
        self.ql.run()
        print(self.vi.mean_discrepancy)
        print(self.pi.mean_discrepancy)
        print(self.ql.mean_discrepancy)
        return
    
class ForestManagementRunner:
    
    @staticmethod
    def runSensitivity():
        for state in [50, 1000]:
            for e in [0.1, 0.01, 0.02, 0.04]:
                for gamma in [0.99, 0.9, 0.85, 0.8]:
                    ForestManagement(state, e, gamma).run()
        
        
class TaxiQ:
    
    def __init__(self, alpha, gamma, eps, eps_decay, conv_eps, conv_counts=300):
        self.env = gym.make('Taxi-v3').unwrapped
        self.Q = np.zeros((500, 6))
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.eps_decay = eps_decay
        self.conv_eps = conv_eps
        self.conv_counts = conv_counts
        self.required_conv_counts = conv_counts
        self.nActions = 6

    def bestAction(self, state):
        return np.argmax(self.Q[state])
    
    def greedyAction(self, state):
        sample = np.random.random()
        if sample <= self.eps:
            randomAction = np.random.randint(self.nActions)
            return randomAction
        else:
            return self.bestAction(state)
    
    def updateQ(self, state0, action, state1, reward):
        Q0 = self.Q[state0][action]
        Q1 = self.Q[state1][np.argmax(self.Q[state1])]
        self.Q[state0][action] = Q0 + self.alpha * (reward + self.gamma * Q1 - Q0)
        return
    
    def isConvergence(self, Q1, Q2):
        diff = np.max(np.abs(Q1 - Q2))
        self.conv_counts = self.conv_counts - 1 if diff <= self.conv_eps else self.required_conv_counts
        return self.conv_counts == 0, diff
    
    def learnPerEpisode(self):
        state = self.env.reset()
        done = False
        while not done:
            action = self.greedyAction(state)
            state1, reward, done, _ = self.env.step(action)
            self.updateQ(state, action, state1, reward)
            state = state1
            print(self.env.render())
        return
    
    def learn(self):
        converged = False
        nEpisode = 0
        while not converged:
            Q1 = copy.deepcopy(self.Q)
            self.learnPerEpisode()
            converged, diff = self.isConvergence(Q1, self.Q)
            nEpisode += 1
            self.eps = self.eps * self.eps_decay
            print(diff, self.eps, self.Q[46, 3], self.Q[462, 4], self.Q[398, 3], self.Q[253, 0])
            
class TaxiQMDP(TaxiQ):
    
    def __init__(self, alpha, gamma, eps, eps_decay, conv_eps, gammaNew, conv_counts=300):
        super().__init__(alpha, gamma, eps, eps_decay, conv_eps, conv_counts)
        self.gammaNew = gammaNew
    
    @staticmethod
    def generateMatrix():
        env = gym.make('Taxi-v3')
        num_states = env.observation_space.n
        num_actions = env.action_space.n
        max_iterations = 1000
        delta = 10**-3
        
        R = np.zeros([num_states, num_actions, num_states])
        T = np.zeros([num_states, num_actions, num_states])
        Q = np.zeros([env.observation_space.n, env.action_space.n])
        gamma = 0.9
        
        print(env.env.desc)
        
        for state in range(num_states):
            for action in range(num_actions):
                for transition in env.env.P[state][action]:
                    probability, next_state, reward, done = transition
                    R[state, action, next_state] = reward
                    T[state, action, next_state] = probability
                T[state, action, :] /= np.sum(T[state, action, :])
        
        value_fn = np.zeros([num_states])
        
        for i in range(max_iterations):
            previous_value_fn = value_fn.copy()
            #learn more about einsum
            Q = np.einsum('ijk,ijk -> ij', T, R + gamma * value_fn)
            value_fn = np.max(Q, axis=1)
            if np.max(np.abs(value_fn - previous_value_fn)) < delta:
                    break                
    
if __name__ == '__main__':
    alpha = 0.07
    gamma = 0.9
    eps = 1.1
    eps_decay = 0.9999
    conv_eps = 0.0000001
    
    taxiQlearner = TaxiQ(alpha, gamma, eps, eps_decay, conv_eps)
    taxiQlearner.learn()
    

        
