# Task 2: Reinforcement Learning Green Food Collection

This implementation provides **4 different reinforcement learning approaches** for robot green food collection using computer vision and DQN:

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

### Unified Training and Testing Scripts

**Important Note**: This project uses unified scripts for all RL methods to ensure consistency and maintainability:

- **`train_rl.py`** - Unified training script for ALL RL methods (DQN, Q-Learning, Policy Gradient, Actor-Critic)
- **`test_rl.py`** - Unified testing script for basic functionality verification
- **`test_actions.py`** - Main simulation and evaluation script

**All training and testing should be done through these unified scripts only.** Legacy per-method scripts have been removed to prevent confusion and ensure consistent behavior across all RL algorithms.

### Training Options

1. **Start CoppeliaSim with Task 2 Scene**
```bash
# Navigate to project directory
cd /path/to/learning_machines_robobo/examples/full_project_setup

# Start CoppeliaSim with arena_approach.ttt scene 
# This scene includes 7 green food objects and Lua script for random food placement
zsh ./scripts/start_coppelia_sim.zsh ./scenes/arena_approach.ttt
```

**Important**: The `arena_approach.ttt` scene includes:
- 7 green food objects (task requirement)
- Lua script (`lua_scripts/arena/food_random.lua`) that automatically randomizes food positions when all foods are collected
- Arena boundaries and obstacles for realistic training environment

2. **Train RL Agents for Food Collection**

#### Train in Simulation
```bash
# Train DQN agent for food collection (recommended)
zsh ./scripts/run_apple_sillicon.zsh train dqn --episodes 100 --max-steps 1800

# Train Q-Learning agent
zsh ./scripts/run_apple_sillicon.zsh train qlearning --episodes 200 --max-steps 1800

# Train Policy Gradient agent
zsh ./scripts/run_apple_sillicon.zsh train policy_gradient --episodes 150 --max-steps 1800

# Train Actor-Critic agent  
zsh ./scripts/run_apple_sillicon.zsh train actor_critic --episodes 100 --max-steps 1800
```

#### Train on Hardware (if available)
```bash
# Train DQN agent on hardware
zsh ./scripts/run_apple_sillicon.zsh train-hw dqn --episodes 50 --max-steps 1800

# Train Q-Learning agent on hardware
zsh ./scripts/run_apple_sillicon.zsh train-hw qlearning --episodes 100 --max-steps 1800
```

3. **Evaluate Trained Models**
```bash
# Evaluate a saved DQN model
zsh ./scripts/run_apple_sillicon.zsh train dqn --mode evaluate --load-model /root/results/figures/rl_model_dqn_1749813555.pth --episodes 10

# Evaluate Q-Learning model
zsh ./scripts/run_apple_sillicon.zsh train qlearning --mode evaluate --load-model /root/results/figures/rl_model_qlearning_1749813555.pkl --episodes 10

# Evaluate on hardware
zsh ./scripts/run_apple_sillicon.zsh train-hw dqn --mode evaluate --load-model /root/results/figures/rl_model_dqn_1749813555.pth --episodes 5
```

4. **Demo All Methods (Quick Comparison)**
```bash
# Train all 4 methods with short episodes for comparison
zsh ./scripts/run_apple_sillicon.zsh train dqn --episodes 10 --max-steps 600
zsh ./scripts/run_apple_sillicon.zsh train qlearning --episodes 20 --max-steps 600
zsh ./scripts/run_apple_sillicon.zsh train policy_gradient --episodes 15 --max-steps 600
zsh ./scripts/run_apple_sillicon.zsh train actor_critic --episodes 10 --max-steps 600
```

## 🎯 Environment & Rewards

### State Space (13 dimensions) - Task 2: Food Collection
- **IR Sensors**: 8 proximity sensors (normalized 0-1)
  - BackL, BackR, FrontL, FrontR, FrontC, FrontRR, BackC, FrontLL
- **Vision Features**: 3 food detection features (normalized 0-1)
  - Distance to nearest green food
  - Angle to nearest green food  
  - Number of detected food objects
- **Robot Orientation**: 2 orientation features (normalized -1 to 1)
  - Pan angle (left/right rotation)
  - Tilt angle (up/down rotation)

### Action Space (9 discrete actions)
- `0`: Stop (0, 0)
- `1`: Backward (-50, -50)
- `2`: Sharp left turn (-25, 25)
- `3`: Left turn (-10, 50)
- `4`: Slight left (0, 50)
- `5`: Forward (50, 50) 
- `6`: Slight right (50, 0)
- `7`: Right turn (50, -10)
- `8`: Sharp right turn (25, -25)

### Reward Function - Task 2: Green Food Collection
- **+1000**: Food collection (primary objective)
- **+100**: Moving towards detected green food
- **+50**: Green food detection in vision
- **+10**: Exploration bonus (visiting new areas)
- **-50**: Collision with obstacles/walls
- **-20**: Moving away from detected food
- **-10**: Timeout penalty (per step)
- **+5000**: Mission completion (collect 7 green foods within 3 minutes)
- **-1000**: Mission failure (time limit exceeded)

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
# Navigate to project directory first
cd /path/to/learning_machines_robobo/examples/full_project_setup

# Start CoppeliaSim with Task 2 scene
zsh ./scripts/start_coppelia_sim.zsh ./scenes/arena_approach.ttt

# Full DQN training in simulation (recommended)
zsh ./scripts/run_apple_sillicon.zsh train dqn --episodes 100 --max-steps 1800

# Quick training for testing
zsh ./scripts/run_apple_sillicon.zsh train dqn --episodes 10 --max-steps 600

# Run on hardware (if available)  
zsh ./scripts/run_apple_sillicon.zsh train-hw dqn --episodes 50 --max-steps 1800

# Evaluate trained model
zsh ./scripts/run_apple_sillicon.zsh train dqn --mode evaluate --load-model /root/results/figures/rl_model_dqn_1749813555.pth --episodes 5
```

## 🔧 Hyperparameter Tuning

### Q-Learning
```python
agent = QLearningAgent(
    state_size=13,          # Task 2: 8 IR + 3 vision + 2 orientation
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
    state_size=13,          # Task 2: 8 IR + 3 vision + 2 orientation
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
    state_size=13,          # Task 2: 8 IR + 3 vision + 2 orientation
    action_size=9,
    learning_rate=0.001,    # Policy network learning rate
    gamma=0.95              # Discount factor for returns
)
```

### Actor-Critic
```python
agent = ActorCriticAgent(
    state_size=13,          # Task 2: 8 IR + 3 vision + 2 orientation
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
- **Success Rate**: 70-90% (collecting 7 foods within time limit)
- **Average Foods Collected**: 5-7 per episode
- **Average Completion Time**: 120-180 seconds
- **Food Detection Accuracy**: 85-95% (green object recognition)

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
# Create custom environment for food collection
env = RobotEnvironment(robot, max_episode_steps=1800)  # 3 minutes at 10 Hz

# Custom reward function for food collection
def custom_reward(self, action_idx, state):
    # Implement food-specific reward logic
    return reward, info
```

### Transfer Learning
```python
# Load pre-trained model and continue training
agent = create_rl_agent('dqn', state_size=12, action_size=9)
agent.load_model('pretrained_food_collection_model.pth')

# Continue training for food collection
train_rl_agent(robot, agent_type='dqn', num_episodes=50)
```

### Multi-Agent Training
```python
# Train multiple agents simultaneously for food collection
agents = [create_rl_agent(method, 12, 9) for method in ['qlearning', 'dqn']]
# Implement multi-agent food collection training loop
```

## 📚 References

- **Q-Learning**: Watkins & Dayan (1992)
- **DQN**: Mnih et al. (2015) - Human-level control through deep reinforcement learning  
- **REINFORCE**: Williams (1992) - Simple statistical gradient-following algorithms
- **Actor-Critic**: Sutton & Barto (2018) - Reinforcement Learning: An Introduction

## 🎯 Next Steps

1. **Computer Vision Enhancement**: Improve green food detection accuracy with better masking
2. **Reality Gap Mitigation**: Develop dual masking strategies for sim-to-real transfer  
3. **Multi-Modal RL**: Combine camera vision with IR sensors for better navigation
4. **Hierarchical RL**: Separate high-level food search from low-level movement control
5. **Curriculum Learning**: Start with fewer foods, progressively increase difficulty
6. **Sim-to-Real Transfer**: Fine-tune simulation models on real robot hardware
