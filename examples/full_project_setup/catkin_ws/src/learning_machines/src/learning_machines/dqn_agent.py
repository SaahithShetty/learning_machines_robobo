"""
Deep Q-Network (DQN) Agent for Task 2: Green Food Collection
============================================================

This module implements the DQN agent and neural network architecture
optimized for the green food collection task.

Components:
- DQNNetwork: Neural network architecture for Q-value approximation
- DQNAgent: Agent implementing experience replay and target networks
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque, namedtuple

# Experience tuple for replay buffer
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


class DQNNetwork(nn.Module):
    """Deep Q-Network architecture matching working reference"""
    
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, action_size)
        # Remove dropout - reference doesn't use it
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)  # No activation on output
        return x


class DQNAgent:
    """Deep Q-Network agent for Task 2: Green Food Collection"""
    
    def __init__(self, state_size: int, action_size: int, learning_rate: float = 0.001,
                 epsilon: float = 1.0, epsilon_decay: float = 0.995, epsilon_min: float = 0.01,
                 gamma: float = 0.95, memory_size: int = 10000, batch_size: int = 32,
                 target_update_freq: int = 100):
        self.state_size = state_size
        self.action_size = action_size
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # Neural networks
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = DQNNetwork(state_size, action_size).to(self.device)
        self.target_network = DQNNetwork(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Experience replay
        self.memory = deque(maxlen=memory_size)
        
        # Training metrics
        self.training_rewards = []
        self.training_losses = []
        self.update_count = 0
        
        # Copy weights to target network
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def get_action(self, state, training=True):
        """Select action using epsilon-greedy policy"""
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.action_size)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.q_network(state_tensor)
        return q_values.argmax().item()
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.append(Experience(state, action, reward, next_state, done))
    
    def update(self, state, action, reward, next_state, done):
        """Store experience and train if enough samples"""
        self.remember(state, action, reward, next_state, done)
        
        if len(self.memory) >= self.batch_size:
            self._train()
    
    def _train(self):
        """Train the network on a batch of experiences"""
        # Sample batch
        batch = random.sample(self.memory, self.batch_size)
        
        # Convert to numpy arrays first, then to tensors for efficiency
        states = torch.FloatTensor(np.array([e.state for e in batch])).to(self.device)
        actions = torch.LongTensor(np.array([e.action for e in batch])).to(self.device)
        rewards = torch.FloatTensor(np.array([e.reward for e in batch])).to(self.device)
        next_states = torch.FloatTensor(np.array([e.next_state for e in batch])).to(self.device)
        dones = torch.BoolTensor(np.array([e.done for e in batch])).to(self.device)
        
        # Current Q-values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Target Q-values
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Update target network
        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Store loss
        self.training_losses.append(loss.item())
    
    def act(self, state, training=True):
        """Alias for get_action for compatibility"""
        return self.get_action(state, training)
    
    def learn(self, state, action, reward, next_state, done):
        """Alias for update for compatibility"""
        return self.update(state, action, reward, next_state, done)
    
    def save_model(self, filepath):
        """Save model to file"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_rewards': self.training_rewards,
            'training_losses': self.training_losses
        }, filepath)
    
    def load_model(self, filepath):
        """Load model from file"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon_min)
        self.training_rewards = checkpoint.get('training_rewards', [])
        self.training_losses = checkpoint.get('training_losses', [])
