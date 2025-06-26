"""
Q-Learning Agent for Task 3: Object Pushing
============================================
Q-Learning Agent for Task 2: Green Food Collection
=================================================

This module implements a tabular Q-Learning agent using discretized states
for the green food collection task.

Components:
- QLearningAgent: Tabular Q-learning with epsilon-greedy exploration
"""

import numpy as np
import pickle
from typing import Tuple


class QLearningAgent:
    """Q-Learning agent with discretized state space for Task 3"""
    
    def __init__(self, state_size: int, action_size: int, learning_rate: float = 0.1, 
                 epsilon: float = 1.0, epsilon_decay: float = 0.995, epsilon_min: float = 0.01,
                 gamma: float = 0.95):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.gamma = gamma
        
        self.q_table = {}
        self.training_rewards = []
        self.training_losses = []
    
    def _discretize_state(self, state) -> Tuple[int, ...]:
        """Convert continuous state to discrete bins"""
        bins = 10
        discrete_state = []
        
        for i, val in enumerate(state):
            val_clipped = np.clip(val, 0.0, 1.0)
            bin_idx = min(int(val_clipped * bins), bins - 1)
            discrete_state.append(bin_idx)
        
        return tuple(discrete_state)
    
    def get_action(self, state, training=True):
        """Select action using epsilon-greedy policy"""
        discrete_state = self._discretize_state(state)
        
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.action_size)
        
        if discrete_state not in self.q_table:
            self.q_table[discrete_state] = np.zeros(self.action_size)
        
        return np.argmax(self.q_table[discrete_state])
    
    def update(self, state, action, reward, next_state, done):
        """Update Q-table using Q-learning update rule"""
        discrete_state = self._discretize_state(state)
        discrete_next_state = self._discretize_state(next_state)
        
        if discrete_state not in self.q_table:
            self.q_table[discrete_state] = np.zeros(self.action_size)
        if discrete_next_state not in self.q_table:
            self.q_table[discrete_next_state] = np.zeros(self.action_size)
        
        current_q = self.q_table[discrete_state][action]
        if done:
            target_q = reward
        else:
            target_q = reward + self.gamma * np.max(self.q_table[discrete_next_state])
        
        self.q_table[discrete_state][action] += self.learning_rate * (target_q - current_q)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        loss = abs(target_q - current_q)
        self.training_losses.append(loss)
    
    def save_model(self, filepath):
        """Save Q-table to file"""
        model_data = {
            'q_table': self.q_table,
            'epsilon': self.epsilon,
            'training_rewards': self.training_rewards,
            'training_losses': self.training_losses
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath):
        """Load Q-table from file"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        self.q_table = model_data['q_table']
        self.epsilon = model_data.get('epsilon', self.epsilon_min)
        self.training_rewards = model_data.get('training_rewards', [])
        self.training_losses = model_data.get('training_losses', [])
