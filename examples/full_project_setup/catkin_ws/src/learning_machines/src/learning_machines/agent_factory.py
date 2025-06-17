"""
Agent Factory for Task 2: Green Food Collection
===============================================

This module provides a factory function to create different types of RL agents
for the green food collection task.

Supported Agents:
- DQN: Deep Q-Network with experience replay
- Q-Learning: Tabular Q-learning with discretized states
- Policy Gradient: REINFORCE algorithm
- Actor-Critic: A2C with shared network
"""

from .dqn_agent import DQNAgent
from .qlearning_agent import QLearningAgent
from .policy_gradient_agent import PolicyGradientAgent
from .actor_critic_agent import ActorCriticAgent


def create_rl_agent(agent_type: str, state_size: int, action_size: int, **kwargs):
    """Factory function to create RL agents for Task 2
    
    Args:
        agent_type: Type of agent ('dqn', 'qlearning', 'policy_gradient', 'actor_critic')
        state_size: Dimension of state space (13 for Task 2)
        action_size: Number of actions (9 for Task 2)
        **kwargs: Additional agent-specific parameters
        
    Returns:
        Initialized RL agent instance
        
    Raises:
        ValueError: If agent_type is not supported
    """
    agents = {
        'dqn': DQNAgent,
        'qlearning': QLearningAgent,
        'policy_gradient': PolicyGradientAgent,
        'actor_critic': ActorCriticAgent
    }
    
    if agent_type.lower() not in agents:
        supported_types = ', '.join(agents.keys())
        raise ValueError(f"Unsupported agent type: {agent_type}. "
                        f"Supported types: {supported_types}")
    
    return agents[agent_type.lower()](state_size, action_size, **kwargs)


def get_agent_info(agent_type: str) -> dict:
    """Get information about a specific agent type
    
    Args:
        agent_type: Type of agent
        
    Returns:
        Dictionary with agent information
    """
    agent_info = {
        'dqn': {
            'name': 'Deep Q-Network',
            'type': 'Value-based',
            'description': 'Uses neural network to approximate Q-values with experience replay',
            'best_for': 'Complex state spaces, stable learning',
            'memory_efficient': False,
            'convergence': 'Fast'
        },
        'qlearning': {
            'name': 'Q-Learning',
            'type': 'Value-based',
            'description': 'Tabular Q-learning with discretized state space',
            'best_for': 'Simple environments, interpretable policies',
            'memory_efficient': True,
            'convergence': 'Guaranteed (tabular)'
        },
        'policy_gradient': {
            'name': 'Policy Gradient (REINFORCE)',
            'type': 'Policy-based',
            'description': 'Direct policy optimization using policy gradients',
            'best_for': 'Continuous action spaces, stochastic policies',
            'memory_efficient': True,
            'convergence': 'Slow but stable'
        },
        'actor_critic': {
            'name': 'Actor-Critic (A2C)',
            'type': 'Actor-Critic',
            'description': 'Combines policy and value learning with advantage estimation',
            'best_for': 'Balanced exploration/exploitation, sample efficiency',
            'memory_efficient': True,
            'convergence': 'Fast and stable'
        }
    }
    
    return agent_info.get(agent_type.lower(), {})


def get_default_hyperparameters(agent_type: str) -> dict:
    """Get default hyperparameters for each agent type optimized for Task 2
    
    Args:
        agent_type: Type of agent
        
    Returns:
        Dictionary with default hyperparameters
    """
    defaults = {
        'dqn': {
            'learning_rate': 0.001,
            'epsilon': 1.0,
            'epsilon_decay': 0.995,
            'epsilon_min': 0.01,
            'gamma': 0.95,
            'memory_size': 10000,
            'batch_size': 32,
            'target_update_freq': 100
        },
        'qlearning': {
            'learning_rate': 0.1,
            'epsilon': 1.0,
            'epsilon_decay': 0.995,
            'epsilon_min': 0.01,
            'gamma': 0.95
        },
        'policy_gradient': {
            'learning_rate': 0.001,
            'gamma': 0.95
        },
        'actor_critic': {
            'learning_rate': 0.001,
            'gamma': 0.95,
            'value_loss_coef': 0.5,
            'entropy_coef': 0.01
        }
    }
    
    return defaults.get(agent_type.lower(), {})
