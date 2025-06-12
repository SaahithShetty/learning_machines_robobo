# Task 1: Reinforcement Learning Obstacle Avoidance

This implementation provides **4 different reinforcement learning approaches** for robot obstacle avoidance:

## 🧠 Available RL Methods

### 1. Q-Learning (Tabular)
- **Type**: Model-free, off-policy
- **State Space**: Discretized IR sensors + orientation
- **Best For**: Simple environments, educational purposes
- **Pros**: Simple, interpretable, guaranteed convergence
- **Cons**: Limited to discrete states, doesn't scale well

### 2. Deep Q-Network (DQN)
- **Type**: Deep RL, experience replay, target network
- **State Space**: Continuous IR sensors + orientation  
- **Best For**: Complex environments, high-dimensional states
- **Pros**: Handles continuous states, stable training
- **Cons**: Requires more data, computationally intensive

### 3. Policy Gradient (REINFORCE)
- **Type**: Policy-based, on-policy
- **State Space**: Continuous IR sensors + orientation
- **Best For**: Continuous action spaces, stochastic policies
- **Pros**: Direct policy optimization, handles continuous actions
- **Cons**: High variance, requires careful tuning

### 4. Actor-Critic (A2C)
- **Type**: Hybrid actor-critic, advantage estimation
- **State Space**: Continuous IR sensors + orientation
- **Best For**: Balance of sample efficiency and stability
- **Pros**: Lower variance than policy gradient, faster than DQN
- **Cons**: More complex, requires tuning multiple networks

## 🚀 Quick Start

### Train Single RL Method
```bash
# Train Q-Learning agent
python rl_controller.py --simulation --rl --agent qlearning --episodes 100

# Train DQN agent  
python rl_controller.py --simulation --rl --agent dqn --episodes 100

# Train Policy Gradient agent
python rl_controller.py --simulation --rl --agent policy_gradient --episodes 100

# Train Actor-Critic agent
python rl_controller.py --simulation --rl --agent actor_critic --episodes 100
```

### Demo All Methods (Quick)
```bash
# Train all 4 methods with short episodes for comparison
python rl_controller.py --simulation --demo-all
```

### Evaluate Trained Model
```bash
# Evaluate a saved model
python rl_controller.py --simulation --rl --agent dqn --mode evaluate --model-path results/rl_model_dqn_1234567890.pth
```

### Full Training + Evaluation
```bash
# Train and then evaluate
python rl_controller.py --simulation --rl --agent actor_critic --mode train_and_evaluate --episodes 200
```

## 🎯 Environment & Rewards

### State Space (11 dimensions)
- **IR Sensors**: 8 proximity sensors (normalized 0-1)
- **Orientation**: sin(yaw), cos(yaw) for continuous representation  
- **Previous Action**: Encoded previous action (0-1)

### Action Space (9 discrete actions)
- `0`: Backward (-50, -50)
- `1`: Sharp left turn (-25, 25)
- `2`: Left turn (-10, 50)
- `3`: Slight left (0, 50)
- `4`: Forward (50, 50) 
- `5`: Slight right (50, 0)
- `6`: Right turn (50, -10)
- `7`: Sharp right turn (25, -25)
- `8`: Stop (0, 0)

### Reward Function
- **+10 × distance**: Reward for forward movement
- **+2.0**: Clear path exploration
- **-5.0**: Near miss (close to obstacle)
- **-50.0**: Collision penalty
- **-1.0**: Unnecessary turning when clear
- **-0.5**: Stopping penalty

## 📊 Training Metrics

Each method saves:
- **Training curves**: Episode rewards, lengths, collision rates
- **Model checkpoints**: Trained neural networks/Q-tables
- **Evaluation results**: Performance on test episodes

Files saved to `results/figures/`:
- `rl_model_{method}_{timestamp}.pth` - Trained model
- `rl_metrics_{method}_{timestamp}.json` - Training data
- `rl_training_{method}_{timestamp}.png` - Training plots

## 🏃‍♂️ Running in Docker

```bash
# Full RL training in simulation
./scripts/run_apple_sillicon.zsh --simulation --rl --agent dqn --episodes 100

# Demo all methods
./scripts/run_apple_sillicon.zsh --simulation --demo-all

# Run on hardware (if available)
./scripts/run_apple_sillicon.zsh --hardware --rl --agent qlearning --episodes 50
```

## 🔧 Hyperparameter Tuning

### Q-Learning
```python
agent = QLearningAgent(
    state_size=9, 
    action_size=9,
    learning_rate=0.1,      # α: learning rate
    epsilon=1.0,            # ε: exploration rate  
    epsilon_decay=0.995,    # ε decay per episode
    epsilon_min=0.01,       # minimum ε
    gamma=0.95              # γ: discount factor
)
```

### DQN
```python
agent = DQNAgent(
    state_size=9,
    action_size=9, 
    learning_rate=0.001,    # Neural network learning rate
    epsilon=1.0,            # ε-greedy exploration
    gamma=0.95,             # Discount factor
    memory_size=10000,      # Experience replay buffer size
    batch_size=32,          # Training batch size
    target_update_freq=100  # Target network update frequency
)
```

### Policy Gradient
```python
agent = PolicyGradientAgent(
    state_size=9,
    action_size=9,
    learning_rate=0.001,    # Policy network learning rate
    gamma=0.95              # Discount factor for returns
)
```

### Actor-Critic
```python
agent = ActorCriticAgent(
    state_size=9,
    action_size=9,
    learning_rate=0.001,    # Learning rate for both networks
    gamma=0.95,             # Discount factor
    value_loss_coef=0.5,    # Critic loss weight
    entropy_coef=0.01       # Entropy bonus for exploration
)
```

## 📈 Expected Performance

### Training Time (100 episodes)
- **Q-Learning**: ~5-10 minutes
- **DQN**: ~15-30 minutes  
- **Policy Gradient**: ~10-20 minutes
- **Actor-Critic**: ~15-25 minutes

### Sample Efficiency
1. **Actor-Critic**: Best sample efficiency
2. **DQN**: Good with experience replay
3. **Q-Learning**: Moderate (discrete states)
4. **Policy Gradient**: Requires more samples

### Final Performance
- **Success Rate**: 70-90% (no collisions)
- **Average Episode Length**: 200-400 steps
- **Distance Traveled**: 5-15 meters per episode

## 🐛 Troubleshooting

### Common Issues
1. **Slow Convergence**: Increase episodes, tune learning rate
2. **High Collision Rate**: Adjust reward penalties, increase exploration
3. **Unstable Training**: Lower learning rate, increase batch size (DQN)
4. **Memory Issues**: Reduce network size, decrease memory buffer

### Debug Commands
```bash
# Check training progress
tail -f results/figures/rl_metrics_*.json

# Monitor GPU usage (if available)
nvidia-smi

# Check model size
ls -lh results/figures/rl_model_*.pth
```

## 🔍 Advanced Usage

### Custom Environment
```python
# Create custom environment
env = RobotEnvironment(robot, max_episode_steps=1000)

# Custom reward function
def custom_reward(self, action_idx, state):
    # Your custom reward logic
    return reward, info
```

### Transfer Learning
```python
# Load pre-trained model and continue training
agent = create_rl_agent('dqn', state_size=9, action_size=9)
agent.load_model('pretrained_model.pth')

# Continue training
train_rl_agent(robot, agent_type='dqn', num_episodes=50)
```

### Multi-Agent Training
```python
# Train multiple agents simultaneously
agents = [create_rl_agent(method, 11, 9) for method in ['qlearning', 'dqn']]
# Implement multi-agent training loop
```

## 📚 References

- **Q-Learning**: Watkins & Dayan (1992)
- **DQN**: Mnih et al. (2015) - Human-level control through deep reinforcement learning  
- **REINFORCE**: Williams (1992) - Simple statistical gradient-following algorithms
- **Actor-Critic**: Sutton & Barto (2018) - Reinforcement Learning: An Introduction

## 🎯 Next Steps

1. **Hyperparameter Optimization**: Use optuna or similar for automated tuning
2. **Curriculum Learning**: Start with simple environments, progressively increase difficulty
3. **Multi-Modal RL**: Incorporate camera vision alongside IR sensors
4. **Sim-to-Real Transfer**: Fine-tune simulation models on real robot
5. **Hierarchical RL**: Combine high-level planning with low-level control
