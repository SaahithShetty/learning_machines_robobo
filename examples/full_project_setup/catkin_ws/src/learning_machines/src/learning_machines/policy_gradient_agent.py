"""
Policy Gradient (REINFORCE) Agent for Task 2: Green Food Collection
===================================================================

This module implements a Policy Gradient agent using the REINFORCE algorithm
for the green food collection task.

Components:
- PolicyNetwork: Neural network for policy approximation
- PolicyGradientAgent: REINFORCE agent with episode-based updates
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque


class PolicyNetwork(nn.Module):
    """Policy network for REINFORCE algorithm"""
    
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 256):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, action_size)
        self.dropout = nn.Dropout(0.2)
        
        # Initialize weights for stability
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier/Glorot initialization with bias toward exploration"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                # Slightly bias the final layer toward forward movement (action 5)
                if m == self.fc3:
                    nn.init.constant_(m.bias, 0.1)
                    # Give slight bias to forward action (action 5 in 9-action space)
                    if self.fc3.out_features > 5:  # Ensure action 5 exists
                        m.bias.data[5] = 0.3
                else:
                    nn.init.constant_(m.bias, 0.01)
    
    def forward(self, x):
        # Add input clamping for numerical stability
        x = torch.clamp(x, -10.0, 10.0)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        
        # Use log_softmax for numerical stability, then exp to get probabilities
        logits = self.fc3(x)
        logits = torch.clamp(logits, -10.0, 10.0)  # Prevent extreme values
        log_probs = F.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)
        
        return probs


class PolicyGradientAgent:
    """REINFORCE Policy Gradient agent for Task 2"""
    
    def __init__(self, state_size: int, action_size: int, learning_rate: float = 0.001,
                 gamma: float = 0.95):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        
        # Neural network
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_network = PolicyNetwork(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)
        
        # Episode memory
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []
        self.episode_log_probs = []
        
        # Training metrics
        self.training_rewards = []
        self.training_losses = []
    
    def get_action(self, state, training=True):
        """Select action using policy network"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        probs = self.policy_network(state_tensor)
        
        if training:
            # Sample from probability distribution
            action_dist = torch.distributions.Categorical(probs)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)
            
            # Store for training
            self.episode_log_probs.append(log_prob)
            
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
        """Update policy at the end of an episode using REINFORCE"""
        if len(self.episode_rewards) == 0:
            return
        
        # Calculate discounted rewards
        discounted_rewards = []
        running_reward = 0
        for reward in reversed(self.episode_rewards):
            running_reward = reward + self.gamma * running_reward
            discounted_rewards.insert(0, running_reward)
        
        # Normalize rewards for stability
        discounted_rewards = torch.FloatTensor(discounted_rewards).to(self.device)
        if len(discounted_rewards) > 1:
            discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-8)
        
        # Calculate policy loss
        policy_loss = []
        for log_prob, reward in zip(self.episode_log_probs, discounted_rewards):
            policy_loss.append(-log_prob * reward)
        
        policy_loss = torch.stack(policy_loss).sum()
        
        # Update policy
        self.optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Store training metrics
        self.training_losses.append(policy_loss.item())
        self.training_rewards.append(sum(self.episode_rewards))
        
        # Clear episode memory
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []
        self.episode_log_probs = []
    
    def update(self, state, action, reward, next_state, done):
        """Store experience and update if episode is done"""
        self.remember(state, action, reward)
        
        if done:
            self.update_episode()
    
    def save_model(self, filepath):
        """Save model to file"""
        torch.save({
            'policy_network_state_dict': self.policy_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_rewards': self.training_rewards,
            'training_losses': self.training_losses
        }, filepath)
    
    def load_model(self, filepath):
        """Load model from file"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_network.load_state_dict(checkpoint['policy_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_rewards = checkpoint.get('training_rewards', [])
        self.training_losses = checkpoint.get('training_losses', [])
