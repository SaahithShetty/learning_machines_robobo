"""
Actor-Critic Agent for Task 3: Object Pushing
===============================================
Actor-Critic Agent for Task 2: Green Food Collection
====================================================

This module implements an Actor-Critic (A2C) agent for the green food collection task.
The actor learns the policy while the critic learns the value function.

Components:
- ActorCriticNetwork: Shared network with actor and critic heads
- ActorCriticAgent: A2C agent with advantage-based updates
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque


class ActorCriticNetwork(nn.Module):
    """Actor-Critic network with shared features for Task 3"""
    
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 256):
        super(ActorCriticNetwork, self).__init__()
        
        # Shared feature layers
        self.shared_fc1 = nn.Linear(state_size, hidden_size)
        self.shared_fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.dropout = nn.Dropout(0.2)
        
        # Actor head (policy)
        self.actor_fc = nn.Linear(hidden_size // 2, action_size)
        
        # Critic head (value function)
        self.critic_fc = nn.Linear(hidden_size // 2, 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier/Glorot initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.01)
                
        # Bias actor toward forward movement (action 5)
        if self.actor_fc.out_features > 5:
            self.actor_fc.bias.data[5] = 0.3
    
    def forward(self, x):
        """Forward pass returning both policy probabilities and state value"""
        # Shared feature extraction
        x = torch.clamp(x, -10.0, 10.0)  # Input stability
        x = F.relu(self.shared_fc1(x))
        x = self.dropout(x)
        x = F.relu(self.shared_fc2(x))
        
        # Actor output (policy probabilities)
        actor_logits = self.actor_fc(x)
        actor_logits = torch.clamp(actor_logits, -10.0, 10.0)
        actor_probs = F.softmax(actor_logits, dim=1)
        
        # Critic output (state value)
        critic_value = self.critic_fc(x)
        
        return actor_probs, critic_value


class ActorCriticAgent:
    """Actor-Critic (A2C) agent for Task 3: Object Pushing"""
    
    def __init__(self, state_size: int, action_size: int, learning_rate: float = 0.001,
                 gamma: float = 0.95, value_loss_coef: float = 0.5, entropy_coef: float = 0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        
        # Neural network
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = ActorCriticNetwork(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        
        # Episode memory
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []
        self.episode_values = []
        self.episode_log_probs = []
        
        # Training metrics
        self.training_rewards = []
        self.training_losses = []
    
    def get_action(self, state, training=True):
        """Select action using actor-critic policy"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        probs, value = self.network(state_tensor)
        
        if training:
            # Sample from probability distribution
            action_dist = torch.distributions.Categorical(probs)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)
            
            # Store for training
            self.episode_log_probs.append(log_prob)
            self.episode_values.append(value)
            
            return action.item()
        else:
            # Greedy action for evaluation
            return probs.argmax().item()
    
    def remember(self, state, action, reward):
        """Store experience for episode-based update"""
        self.episode_states.append(state)
        self.episode_actions.append(action)
        self.episode_rewards.append(reward)
    
    def update_episode(self):
        """Update actor-critic at the end of an episode"""
        if len(self.episode_rewards) == 0:
            return
        
        # Calculate returns (discounted rewards)
        returns = []
        running_return = 0
        for reward in reversed(self.episode_rewards):
            running_return = reward + self.gamma * running_return
            returns.insert(0, running_return)
        
        returns = torch.FloatTensor(returns).to(self.device)
        values = torch.stack(self.episode_values).squeeze()
        log_probs = torch.stack(self.episode_log_probs)
        
        # Calculate advantages
        advantages = returns - values
        
        # Normalize advantages for stability
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Calculate losses
        actor_loss = -(log_probs * advantages.detach()).mean()
        critic_loss = F.mse_loss(values, returns)
        
        # Entropy bonus for exploration
        probs = torch.exp(log_probs)
        entropy = -(probs * log_probs).mean()
        
        # Total loss
        total_loss = actor_loss + self.value_loss_coef * critic_loss - self.entropy_coef * entropy
        
        # Update network
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
        self.optimizer.step()
        
        # Store training metrics
        self.training_losses.append(total_loss.item())
        self.training_rewards.append(sum(self.episode_rewards))
        
        # Clear episode memory
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []
        self.episode_values = []
        self.episode_log_probs = []
    
    def update(self, state, action, reward, next_state, done):
        """Store experience and update if episode is done"""
        self.remember(state, action, reward)
        
        if done:
            self.update_episode()
    
    def save_model(self, filepath):
        """Save model to file"""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_rewards': self.training_rewards,
            'training_losses': self.training_losses
        }, filepath)
    
    def load_model(self, filepath):
        """Load model from file"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_rewards = checkpoint.get('training_rewards', [])
        self.training_losses = checkpoint.get('training_losses', [])
