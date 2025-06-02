# 🤖 Machine Learning with Robobo Robot

Welcome to your machine learning development environment for the Robobo robot! This guide will help you implement AI models and behaviors using computer vision, reinforcement learning, and other ML techniques.

## 🚀 Quick Start

### 1. Test Your Setup
First, verify everything works with the system tests:
```bash
# In your Docker container:
cd /workspace/catkin_ws/src/learning_machines/scripts
python ml_robobo_controller.py --simulation --test
```

### 2. Run Your First ML Experiment
```bash
# Start basic ML behavior in simulation:
python ml_robobo_controller.py --simulation --ml
```

### 3. Develop Your Own Models
Edit the ML controller to implement your algorithms:
```bash
# Navigate to your development directory:
cd ../src/learning_machines/
# Edit ml_controller.py with your ML models
```

## 📁 Project Structure

```
learning_machines/
├── scripts/
│   ├── learning_robobo_controller.py  # Original controller
│   └── ml_robobo_controller.py        # New ML controller
└── src/learning_machines/
    ├── __init__.py                     # Package init
    ├── test_actions.py                 # Original robot tests
    └── ml_controller.py                # Your ML development area
```

## 🧠 Available ML Approaches

### 1. Computer Vision
```python
# Access camera data
image = robot.read_image_front()  # RGB image as numpy array

# Example processing
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 150)

# Implement:
# - Object detection (YOLO, etc.)
# - Object tracking
# - Color segmentation
# - Feature extraction
# - Visual navigation
```

### 2. Reinforcement Learning
```python
# Define state space (sensor readings)
state = {
    'ir_sensors': robot.read_irs(),     # 8 proximity sensors
    'orientation': robot.read_orientation(),
    'wheel_position': robot.read_wheels()
}

# Define action space
actions = [(left_speed, right_speed), ...]  # Motor commands

# Implement:
# - Q-Learning
# - Deep Q-Networks (DQN)
# - Policy Gradient methods
# - Actor-Critic algorithms
```

### 3. Sensor Fusion
```python
# Combine multiple sensor modalities
sensor_data = {
    'vision': robot.read_image_front(),
    'proximity': robot.read_irs(),
    'inertial': robot.read_accel(),
    'orientation': robot.read_orientation()
}

# Implement:
# - Kalman filtering
# - Particle filters
# - Sensor fusion networks
# - SLAM algorithms
```

### 4. Behavioral AI
```python
# Emotion-based responses
if obstacle_detected:
    robot.set_emotion(Emotion.WORRIED)
    robot.play_emotion_sound(SoundEmotion.DISCOMFORT)
else:
    robot.set_emotion(Emotion.HAPPY)

# Implement:
# - Finite state machines
# - Behavior trees
# - Reactive behaviors
# - Social robotics
```

## 🛠 Development Workflow

### 1. Create Your ML Model
```python
class MyMLController(MLController):
    def __init__(self, robot):
        super().__init__(robot)
        # Initialize your model
        self.model = self.load_model()
    
    def predict_action(self, sensor_data):
        # Your ML inference here
        return action
    
    def train_step(self, state, action, reward, next_state):
        # Your training logic here
        pass
```

### 2. Test in Simulation
```bash
# Run your model in simulation first
python ml_robobo_controller.py --simulation --ml
```

### 3. Deploy to Hardware
```bash
# When ready, test on real robot
python ml_robobo_controller.py --hardware --ml
```

## 🔧 Robot API Reference

### Sensors (Inputs)
- `read_image_front()` - Camera image (480x640x3 RGB)
- `read_irs()` - IR proximity sensors (8 values, 0-2m range)
- `read_accel()` - Accelerometer (x, y, z)
- `read_orientation()` - Gyroscope (yaw, pitch, roll)
- `read_wheels()` - Wheel encoders (position, speed)
- `read_phone_pan/tilt()` - Camera orientation

### Actions (Outputs)
- `move_blocking(left, right, duration)` - Wheel movement (-100 to 100)
- `set_phone_pan/tilt_blocking(pos, speed)` - Camera movement
- `set_emotion(emotion)` - Display emotions
- `talk(message)` - Text-to-speech
- `set_led(id, color)` - LED control

### Simulation Only
- `get_position()` - World position (x, y, z)
- `set_position(pos, orient)` - Teleport robot
- `get_sim_time()` - Simulation time

## 📊 Data Collection & Logging

### Save Sensor Data
```python
import json
import numpy as np

# Collect data for training
data_log = []
for step in range(1000):
    sensor_data = robot.get_sensor_data()
    action = robot.predict_action(sensor_data)
    
    # Log for later analysis
    data_log.append({
        'step': step,
        'sensors': sensor_data,
        'action': action,
        'timestamp': time.time()
    })

# Save to file
with open('robot_data.json', 'w') as f:
    json.dump(data_log, f)
```

### Visualize Training Progress
```python
import matplotlib.pyplot as plt

# Plot reward over time
plt.plot(rewards)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Training Progress')
plt.savefig('training_progress.png')
```

## 🎯 Example ML Projects

### 1. Obstacle Avoidance
- **Goal**: Navigate without hitting walls
- **Sensors**: IR proximity sensors
- **ML**: Reinforcement learning (Q-learning)
- **Reward**: +1 for forward movement, -10 for collisions

### 2. Object Following
- **Goal**: Follow colored objects
- **Sensors**: Camera
- **ML**: Computer vision + control
- **Method**: Color detection → PID control

### 3. Exploration Behavior
- **Goal**: Explore unknown environments
- **Sensors**: Camera + IR + odometry
- **ML**: Curiosity-driven RL
- **Reward**: Novelty-based rewards

### 4. Emotion Recognition
- **Goal**: Respond to human faces/gestures
- **Sensors**: Camera
- **ML**: Deep learning (CNN)
- **Output**: Emotional responses

## 🐛 Debugging Tips

### Common Issues
1. **Robot not moving**: Check motor speeds are in range (-100, 100)
2. **Camera not working**: Verify `camera=True` in HardwareRobobo
3. **Simulation freezing**: Use `robot.play_simulation()` before starting
4. **Import errors**: Ensure you're in the correct directory

### Debugging Tools
```python
# Print sensor values
sensor_data = robot.get_sensor_data()
print(f"IR sensors: {sensor_data['ir_sensors']}")
print(f"Orientation: {sensor_data['orientation']}")

# Visualize camera
cv2.imshow('Robot Camera', sensor_data['image'])
cv2.waitKey(1)

# Log to file
import logging
logging.basicConfig(filename='robot.log', level=logging.INFO)
logging.info(f"Action taken: {action}")
```

## 🚀 Next Steps

1. **Start Simple**: Begin with basic obstacle avoidance
2. **Iterate**: Gradually add complexity to your models
3. **Validate**: Test extensively in simulation before hardware
4. **Document**: Keep track of what works and what doesn't
5. **Experiment**: Try different ML approaches and compare results

## 💡 Advanced Topics

- **Multi-robot coordination**: Control multiple Robobos
- **Transfer learning**: Pre-trained vision models
- **Sim-to-real transfer**: Bridge simulation and hardware gaps
- **Human-robot interaction**: Voice and gesture recognition
- **SLAM**: Simultaneous localization and mapping

Happy coding! 🤖✨
