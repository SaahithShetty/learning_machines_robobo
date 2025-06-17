"""
Task 2: Green Food Collection - Reinforcement Learning Implementation
====================================================================

This module implements Task 2 for the Robobo robot using reinforcement learning
and computer vision. The robot must collect 7 green food boxes within 3 minutes
using DQN (Deep Q-Network) for intelligent navigation and OpenCV for food detection.

Key Components:
- RobotEnvironment: RL environment wrapper optimized for food collection
- DQNAgent: Deep Q-Network agent for action selection and learning
- FoodVisionProcessor: Computer vision system for green food detection
- Comprehensive reward system balancing collection, exploration, and safety

State Space (13D): [8 IR sensors + 3 vision features + 2 orientation]
Action Space (9D): Stop, backward, turns (left/right), forward movements
Sensor Order: [BackL, BackR, FrontL, FrontR, FrontC, FrontRR, BackC, FrontLL]

Author: Learning Machines Team
Focus: Task 2 (Green Food Collection) - Obstacle Avoidance (Task 1) removed
"""

import cv2
import time
import math
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import random
import pickle
import json
import sys
from collections import deque, namedtuple

# Try to import data_files, fallback to local directory if not available
try:
    from data_files import FIGURES_DIR
except ImportError:
    # Fallback for local development (not in Docker)
    FIGURES_DIR = Path("results/figures")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

from robobo_interface import (
    IRobobo,
    Emotion,
    LedId,
    LedColor,
    SoundEmotion,
    SimulationRobobo,
    HardwareRobobo,
)

from typing import Optional

# Import RL agents from separate modules
try:
    from .agent_factory import create_rl_agent, get_agent_info, get_default_hyperparameters
except ImportError:
    # Fallback for when modules are in same directory
    from agent_factory import create_rl_agent, get_agent_info, get_default_hyperparameters

# Experience tuple for environment
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


def get_emotion_from_reward(reward: float, collision: bool = False, near_miss: bool = False) -> Emotion:
    """Determine robot emotion based on reward and situation"""
    if collision:
        return Emotion.SAD
    elif near_miss:
        return Emotion.SURPRISED
    elif reward > 5.0:
        return Emotion.HAPPY
    elif reward < -2.0:
        return Emotion.ANGRY
    else:
        return Emotion.NORMAL


def print_dynamic_step_info(episode: int, step: int, action: int, reward: float, 
                           emotion: Emotion, ir_sensors: list, info: dict, 
                           episode_reward: float):
    """Print step information using carriage return to overwrite same line"""
    action_names = ["Back", "TurnL", "TurnL_S", "FwdL", "Forward", "FwdR", "TurnR_S", "TurnR", "Stop"]
    action_name = action_names[action] if 0 <= action < len(action_names) else f"A{action}"
    
    # Get front sensor readings (indices 2,3,4 = FrontL, FrontR, FrontC)
    if len(ir_sensors) >= 5:
        front_sensors = [ir_sensors[2], ir_sensors[3], ir_sensors[4]]  # FrontL, FrontR, FrontC
    else:
        front_sensors = [0.0, 0.0, 0.0]
    min_distance = min(front_sensors)
    
    # Create dynamic display string
    status_str = (
        f"Ep:{episode:3d} St:{step:3d} | "
        f"Action:{action_name:8s} | "
        f"Reward:{reward:+6.2f} | "
        f"Emotion:{emotion.value:10s} | "
        f"MinDist:{min_distance:.3f} | "
        f"EpRwd:{episode_reward:+7.2f} | "
        f"Collision:{info.get('collision', False)} | "
        f"NearMiss:{info.get('near_miss', False)}"
    )
    
    # Print with carriage return to overwrite previous line
    print(f"\r{status_str}", end='', flush=True)


def test_emotions(rob: IRobobo):
    rob.set_emotion(Emotion.HAPPY)
    rob.talk("Hello")
    rob.play_emotion_sound(SoundEmotion.PURR)
    rob.set_led(LedId.FRONTCENTER, LedColor.GREEN)


def test_move_and_wheel_reset(rob: IRobobo):
    rob.move_blocking(100, 100, 1000)
    print("before reset: ", rob.read_wheels())
    rob.reset_wheels()
    rob.sleep(1)
    print("after reset: ", rob.read_wheels())


def test_sensors(rob: IRobobo):
    print("IRS data: ", rob.read_irs())
    image = rob.read_image_front()
    cv2.imwrite(str(FIGURES_DIR / "photo.png"), image)
    print("Phone pan: ", rob.read_phone_pan())
    print("Phone tilt: ", rob.read_phone_tilt())
    print("Current acceleration: ", rob.read_accel())
    print("Current orientation: ", rob.read_orientation())


def test_phone_movement(rob: IRobobo):
    rob.set_phone_pan_blocking(20, 100)
    print("Phone pan after move to 20: ", rob.read_phone_pan())
    rob.set_phone_tilt_blocking(50, 100)
    print("Phone tilt after move to 50: ", rob.read_phone_tilt())


def test_sim(rob: SimulationRobobo):
    print("Current simulation time:", rob.get_sim_time())
    print("Is the simulation currently running? ", rob.is_running())
    rob.stop_simulation()
    print("Simulation time after stopping:", rob.get_sim_time())
    print("Is the simulation running after shutting down? ", rob.is_running())
    rob.play_simulation()
    print("Simulation time after starting again: ", rob.get_sim_time())
    print("Current robot position: ", rob.get_position())
    print("Current robot orientation: ", rob.get_orientation())

    pos = rob.get_position()
    orient = rob.get_orientation()
    rob.set_position(pos, orient)
    print("Position the same after setting to itself: ", pos == rob.get_position())
    print("Orient the same after setting to itself: ", orient == rob.get_orientation())


def test_hardware(rob: HardwareRobobo):
    print("Phone battery level: ", rob.phone_battery())
    print("Robot battery level: ", rob.robot_battery())


def run_green_food_collection(rob: IRobobo, rl_agent_type: str = 'dqn', 
                             rl_mode: str = 'train', rl_episodes: int = 100):
    """Main function to run Task 2: Green Food Collection
    
    This is the primary entry point for running the green food collection task
    using reinforcement learning and computer vision.
    
    Args:
        rob: Robot interface instance (SimulationRobobo or HardwareRobobo)
        rl_agent_type: Type of RL agent ('dqn' for Task 2)
        rl_mode: RL mode ('train', 'evaluate', or 'train_and_evaluate')
        rl_episodes: Number of RL episodes to run
        
    Returns:
        Dictionary with training results and performance metrics
    """
    if isinstance(rob, SimulationRobobo):
        rob.play_simulation()
    
    print("="*50)
    print(f"TASK 2: GREEN FOOD COLLECTION USING OPENCV + RL ({rl_agent_type.upper()})")
    print("="*50)
    
    # Run Task 2 - Green Food Collection
    results = green_food_collection_task2(
        rob, agent_type=rl_agent_type, mode=rl_mode, num_episodes=rl_episodes
    )
    
    print(f"\nTask 2 completed!")
    if rl_mode in ['train', 'train_and_evaluate']:
        print(f"Models and metrics saved to {FIGURES_DIR}")
    
    if isinstance(rob, SimulationRobobo):
        rob.stop_simulation()
        
    return results


class RobotEnvironment:
    """Environment wrapper for Task 2: Green Food Collection using Reinforcement Learning
    
    This environment is specifically designed for Task 2, which involves:
    - Detecting and collecting 7 green food boxes within 3 minutes
    - Using computer vision for food detection and localization
    - Implementing intelligent navigation with obstacle avoidance
    - Providing a 13-dimensional state space: [8 IR sensors + 3 vision + 2 orientation]
    - Supporting 9 discrete actions optimized for food collection
    """
    
    def __init__(self, robot: IRobobo, vision_processor, max_episode_steps: int = 1000, 
                 max_episode_time: int = 180):
        self.robot = robot
        self.vision_processor = vision_processor
        self.max_episode_steps = max_episode_steps
        self.max_episode_time = max_episode_time  # 3 minutes for Task 2
        self.step_count = 0
        self.is_simulation = isinstance(robot, SimulationRobobo)
        
        # Action space: Task 2 specific actions with stop capability for food collection
        self.action_space_size = 9
        self.actions = [
            (0, 0),      # 0: Stop (useful for precise food collection)
            (-50, -50),  # 1: Backward
            (-30, 30),   # 2: Turn Left
            (-15, 30),   # 3: Turn Left Slight
            (40, 60),    # 4: Forward Left
            (60, 60),    # 5: Forward
            (60, 40),    # 6: Forward Right
            (30, -15),   # 7: Turn Right Slight
            (30, -30),   # 8: Turn Right
        ]
        
        # Action descriptions for debugging
        self.action_descriptions = [
            "Stop", "Backward", "Turn Left", "Turn Left Slight", "Forward Left", 
            "Forward", "Forward Right", "Turn Right Slight", "Turn Right"
        ]
        
        # State space: IR sensors (8) + vision data (3) + orientation (2) = 13 dimensions
        # Vision data: [food_detected, food_distance, food_angle]
        # Orientation: [yaw_normalized, pitch_normalized]
        self.state_size = 13
        
        # Task 2 specific tracking
        self.food_collected = 0
        self.total_food_target = 7
        self.episode_start_time = None
        self.last_food_positions = []
        
        # Obstacle detection thresholds (for collision avoidance while collecting food)
        if self.is_simulation:
            self.obstacle_threshold = 200  # Distance-based threshold for simulation
        else:
            self.obstacle_threshold = 10   # Intensity-based threshold for hardware
        
        self.reset()
    
    def reset(self):
        """Reset environment for new episode"""
        self.step_count = 0
        
        if self.is_simulation:
            # Reset robot position in simulation
            self.robot.stop_simulation()
            time.sleep(0.5)
            self.robot.play_simulation()
            time.sleep(1.0)
        
        # Reset Task 2 specific counters
        self.food_collected = 0
        self.episode_start_time = time.time()
        self.last_food_positions = []
        
        # Debug: Check initial state
        initial_state = self._get_state()
        ir_raw = self.robot.read_irs()
        print(f"Reset - IR readings: {ir_raw[:4]}, Food collected: {self.food_collected}")
        
        return initial_state
    
    def _get_state(self):
        """Get current state representation for Task 2: [IR sensors + vision + orientation]"""
        # IR sensors (8 dimensions) - Correct order: [BackL, BackR, FrontL, FrontR, FrontC, FrontRR, BackC, FrontLL]
        ir_values = self.robot.read_irs()
        ir_normalized = []
        for val in ir_values:
            if val is None:
                ir_normalized.append(0.0)  # No detection
            else:
                if self.is_simulation:
                    # In simulation, values are distance-based: higher = farther, lower = closer
                    # Normalize to 0-1 where 0 = very close, 1 = far away
                    ir_normalized.append(min(val / 2000.0, 1.0))
                else:
                    # In hardware, values are intensity-based: higher = closer, lower = farther
                    # Normalize to 0-1 where 0 = far away, 1 = very close
                    ir_normalized.append(min(val / 100.0, 1.0))
        
        # Vision data (3 dimensions: food_detected, food_distance, food_angle)
        camera_frame = self.robot.read_image_front()
        food_objects, _ = self.vision_processor.detect_green_food(camera_frame)
        
        if food_objects and len(food_objects) > 0:
            # Use the most confident detection
            best_food = food_objects[0]
            food_detected = 1.0
            food_distance = min(1.0, best_food['distance'] / 3.0)  # Normalize to 0-1
            food_angle = best_food['angle'] / 30.0  # Normalize to -1 to +1
        else:
            food_detected = 0.0
            food_distance = 1.0  # Max distance when no food detected
            food_angle = 0.0
        
        # Orientation (2 dimensions)
        orientation = self.robot.read_orientation()
        if orientation:
            orient_x = orientation.yaw / 180.0  # Normalize to -1 to +1
            orient_y = orientation.pitch / 180.0
        else:
            orient_x, orient_y = 0.0, 0.0
        
        # Combine: 8 IR + 3 vision + 2 orientation = 13 dimensions
        state = np.array(ir_normalized + [food_detected, food_distance, food_angle, orient_x, orient_y], 
                        dtype=np.float32)
        return state
    
    def step(self, action_idx):
        """Execute action and return (next_state, reward, done, info)"""
        self.step_count += 1
        
        # Execute action
        left_speed, right_speed = self.actions[action_idx]
        
        # Debug: Print action being taken
        if self.step_count <= 10 or self.step_count % 50 == 0:
            print(f"Step {self.step_count}: Action {action_idx} ({self.action_descriptions[action_idx]}) - "
                  f"Speeds: L={left_speed}, R={right_speed}")
        
        self.robot.move_blocking(left_speed, right_speed, 200)
        
        # Get new state
        next_state = self._get_state()
        
        # Calculate reward and determine emotion
        reward, info = self._calculate_reward(action_idx, next_state)
        
        # Debug: Print reward info occasionally
        if self.step_count <= 10 or self.step_count % 50 == 0:
            print(f"  Reward: {reward:.2f}, Food collected: {self.food_collected}, Info: {info}")
        
        # Set emotion based on reward and situation
        if info.get('food_collected', False):
            emotion = Emotion.HAPPY
        elif info.get('collision', False):
            emotion = Emotion.SAD
        elif info.get('food_detected', False):
            emotion = Emotion.SURPRISED  # Excited about finding food
        elif reward > 10.0:
            emotion = Emotion.HAPPY
        elif reward < -5.0:
            emotion = Emotion.ANGRY
        else:
            emotion = Emotion.NORMAL
        
        # Store emotion and additional info
        info['emotion'] = emotion
        info['ir_sensors'] = list(next_state[:8])  # Store IR sensor readings as list
        info['vision_data'] = list(next_state[8:11])  # Store vision data as list
        info['action_taken'] = self.action_descriptions[action_idx]
        info['food_collected_count'] = int(self.food_collected)
        
        # Check if episode is done
        time_elapsed = time.time() - self.episode_start_time if self.episode_start_time else 0
        done = (
            self.step_count >= self.max_episode_steps or 
            time_elapsed >= self.max_episode_time or
            self.food_collected >= self.total_food_target
        )
        
        # Update food collection count
        if info.get('food_collected', False):
            self.food_collected += 1
        
        return next_state, reward, done, info
    
    def _calculate_reward(self, action_idx, state):
        """Calculate reward for Task 2: Green Food Collection"""
        info = {
            'collision': False, 
            'food_collected': False, 
            'food_detected': False,
            'ir_sensors': [],
            'vision_data': [],
            'action_taken': '',
            'food_collected_count': 0
        }
        reward = 0.0
        
        # Extract state components
        ir_values = state[:8]  # IR sensor readings
        food_detected = state[8] > 0.5  # Food detection flag
        food_distance = state[9]  # Normalized food distance
        food_angle = state[10]  # Normalized food angle (-1 to +1)
        
        # Time-based calculations
        time_elapsed = time.time() - self.episode_start_time if self.episode_start_time else 0
        time_remaining = max(0, self.max_episode_time - time_elapsed)
        
        # 1. Food Detection Reward
        if food_detected:
            reward += 15  # Reward for seeing food
            info['food_detected'] = True
            
            # 2. Food Approach Progress Reward
            # Distance reward (closer is better)
            distance_reward = (1.0 - food_distance) * 20
            reward += distance_reward
            
            # Alignment reward (facing food is better)
            alignment_reward = (1.0 - abs(food_angle)) * 10
            reward += alignment_reward
        
        # 3. Food Collection Reward (check for collision with food)
        if self._check_food_collision():
            base_reward = 100
            time_bonus = (time_remaining / self.max_episode_time) * 50  # Earlier = better
            reward += base_reward + time_bonus
            info['food_collected'] = True
            
        # 4. Time Pressure Penalty
        time_pressure = 1.0 - (time_remaining / self.max_episode_time)
        time_penalty = -(1.0 + 2.0 * time_pressure)
        reward += time_penalty
        
        # 5. Action-specific rewards and penalties
        if action_idx == 0:  # Stop action
            if food_detected and food_distance < 0.3:
                # Reward stopping near food for precise collection
                reward += 5
            else:
                # Penalty for stopping when not near food
                reward -= 15
        elif action_idx == 5:  # Forward action
            reward += 2  # Base reward for forward movement
        elif action_idx in [1]:  # Backward
            reward += 1  # Small reward for backward movement
        else:  # Turning actions
            reward += 0.5  # Small reward for turning
        
        # 6. Obstacle Avoidance (prevent collisions while collecting food)
        min_ir = min(ir_values)
        if self.is_simulation:
            # In simulation, lower values = closer obstacles
            collision_threshold = self.obstacle_threshold / 2000.0  # Normalize
        else:
            # In hardware, higher values = closer obstacles
            collision_threshold = self.obstacle_threshold / 100.0  # Normalize
        
        if min_ir < collision_threshold:
            reward -= 20
            info['collision'] = True
        
        # 7. Mission Progress Bonus
        if self.food_collected >= 6:  # Almost complete
            reward += 30
        elif self.food_collected >= 4:  # Good progress
            reward += 15
        
        return reward, info
        
    def _check_food_collision(self):
        """Check if robot has collided with food (simplified detection)"""
        # In real implementation, this would be handled by the simulation
        # or detected through vision (food disappearing from view when very close)
        # For now, we'll use a simplified approach based on very close food detection
        
        camera_frame = self.robot.read_image_front()
        food_objects, _ = self.vision_processor.detect_green_food(camera_frame)
        
        if food_objects:
            closest_food = min(food_objects, key=lambda x: x['distance'])
            # If food is very close and centered, consider it collected
            if closest_food['distance'] < 0.2 and abs(closest_food['angle']) < 10:
                return True
                
        return False


# ============================================================================
# RL AGENTS MOVED TO SEPARATE MODULES
# ============================================================================
# 
# The following RL agents have been moved to separate files for better organization:
# - DQNAgent: dqn_agent.py
# - QLearningAgent: qlearning_agent.py  
# - PolicyGradientAgent: policy_gradient_agent.py
# - ActorCriticAgent: actor_critic_agent.py
#
# Use create_rl_agent() from agent_factory.py to instantiate agents
#

# ============================================================================
# TASK 2: GREEN FOOD COLLECTION USING OPENCV + RL
# ============================================================================

# DQN Network and Agent classes have been moved to dqn_agent.py
# These classes include:
# - DQNNetwork: Deep Q-Network architecture with fully connected layers
# - DQNAgent: DQN agent with experience replay, target network, and epsilon-greedy exploration


# Policy Network and Policy Gradient Agent classes have been moved to policy_gradient_agent.py
# These classes include:
# - PolicyNetwork: Neural network for REINFORCE with exploration bias
# - PolicyGradientAgent: REINFORCE policy gradient agent with context-aware action filtering


# Actor-Critic Network and Agent classes have been moved to actor_critic_agent.py
# These classes include:
# - ActorCriticNetwork: Neural network with shared features for actor-critic algorithm
# - ActorCriticAgent: A2C agent with value function and policy optimization


# Q-Learning Agent class has been moved to qlearning_agent.py
# This class includes:
# - QLearningAgent: Q-Learning agent with epsilon-greedy exploration and discrete state space


# Agent factory function has been moved to agent_factory.py
# Use create_rl_agent() function for agent instantiation

# ============================================================================
# TASK 2: GREEN FOOD COLLECTION USING OPENCV + DQN
# ============================================================================

class FoodVisionProcessor:
    """Computer vision processor for green food detection with dual masking"""
    
    def __init__(self, environment_type="simulation"):
        self.environment_type = environment_type
        self.setup_environment_config()
        
    def setup_environment_config(self):
        """Configure parameters based on environment (simulation vs hardware)"""
        if self.environment_type == "simulation":
            # Simulation: More predictable lighting, cleaner colors
            self.green_ranges = {
                'primary': {'lower': np.array([40, 60, 60]), 'upper': np.array([80, 255, 255])},
                'backup': {'lower': np.array([35, 40, 40]), 'upper': np.array([85, 255, 255])}
            }
            self.morphology_kernels = {
                'opening': cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
                'closing': cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
                'dilation': cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
            }
            self.min_area = 50
            self.max_area_ratio = 0.15
            self.distance_calibration = {'reference_area': 1000, 'reference_distance': 1.0}
            
        else:  # Hardware
            # Hardware: Variable lighting, noisy images, need robust detection
            self.green_ranges = {
                'primary': {'lower': np.array([30, 30, 30]), 'upper': np.array([90, 255, 255])},
                'backup': {'lower': np.array([25, 20, 20]), 'upper': np.array([95, 255, 255])}
            }
            self.morphology_kernels = {
                'opening': cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
                'closing': cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
                'dilation': cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            }
            self.min_area = 80
            self.max_area_ratio = 0.25
            self.distance_calibration = {'reference_area': 800, 'reference_distance': 1.0}
            
        # Common parameters
        self.camera_fov = 60  # degrees
        self.confidence_threshold = 0.7
        
    def detect_green_food(self, camera_frame):
        """
        Detect green food objects using dual masking strategy
        
        Args:
            camera_frame: BGR image from camera
            
        Returns:
            food_objects: List of detected food objects with position/distance info
            debug_mask: Binary mask for visualization (optional)
        """
        if camera_frame is None or camera_frame.size == 0:
            return [], np.zeros((480, 640), dtype=np.uint8)
            
        # Step 1: Preprocessing
        preprocessed = self.preprocess_image(camera_frame)
        
        # Step 2: Color space conversion
        hsv = cv2.cvtColor(preprocessed, cv2.COLOR_BGR2HSV)
        
        # Step 3: Dual masking approach
        primary_mask = self.create_primary_mask(hsv)
        backup_mask = self.create_backup_mask(hsv)
        
        # Step 4: Mask fusion and cleaning
        combined_mask = self.fuse_and_clean_masks(primary_mask, backup_mask)
        
        # Step 5: Contour detection and filtering
        food_objects = self.detect_and_filter_contours(combined_mask, camera_frame)
        
        return food_objects, combined_mask
        
    def preprocess_image(self, image):
        """Preprocess image based on environment type"""
        if self.environment_type == "hardware":
            # Hardware: More aggressive preprocessing
            # Denoise
            denoised = cv2.bilateralFilter(image, 9, 75, 75)
            # Enhance contrast
            lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8)).apply(l)
            enhanced = cv2.merge([l, a, b])
            return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        else:
            # Simulation: Minimal preprocessing
            return cv2.GaussianBlur(image, (3, 3), 0)
            
    def create_primary_mask(self, hsv_image):
        """Create primary green mask with strict parameters"""
        lower = self.green_ranges['primary']['lower']
        upper = self.green_ranges['primary']['upper']
        return cv2.inRange(hsv_image, lower, upper)
        
    def create_backup_mask(self, hsv_image):
        """Create backup green mask with relaxed parameters"""
        lower = self.green_ranges['backup']['lower']
        upper = self.green_ranges['backup']['upper']
        return cv2.inRange(hsv_image, lower, upper)
        
    def fuse_and_clean_masks(self, primary_mask, backup_mask):
        """Fuse masks and apply morphological operations"""
        # Combine masks: Primary mask gets priority, backup fills gaps
        combined = cv2.bitwise_or(primary_mask, backup_mask)
        
        # Morphological operations to clean up the mask
        # Remove small noise
        cleaned = cv2.morphologyEx(combined, cv2.MORPH_OPEN, self.morphology_kernels['opening'])
        # Fill gaps in objects
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, self.morphology_kernels['closing'])
        # Slightly expand to ensure full object coverage
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_DILATE, self.morphology_kernels['dilation'])
        
        return cleaned
        
    def detect_and_filter_contours(self, mask, original_image):
        """Detect contours and filter for food-like objects"""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        food_objects = []
        height, width = original_image.shape[:2]
        max_area = height * width * self.max_area_ratio
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Size filtering
            if area < self.min_area or area > max_area:
                continue
                
            # Shape filtering
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h
            
            # Food boxes should have reasonable aspect ratio
            if aspect_ratio < 0.3 or aspect_ratio > 3.0:
                continue
                
            # Solidity filtering (reject very irregular shapes)
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            
            if solidity < 0.6:  # Reject very irregular shapes
                continue
                
            # Edge proximity filtering (avoid partial objects)
            margin = 15
            if x < margin or y < margin or x + w > width - margin or y + h > height - margin:
                continue
                
            # Calculate object properties
            center_x, center_y = x + w // 2, y + h // 2
            distance = self.estimate_distance(area)
            angle = self.calculate_angle(center_x, width)
            confidence = self.calculate_confidence(contour, area, solidity)
            
            food_objects.append({
                'center': (center_x, center_y),
                'area': area,
                'distance': distance,
                'angle': angle,
                'confidence': confidence,
                'bbox': (x, y, w, h)
            })
            
        # Sort by confidence (best detections first)
        food_objects.sort(key=lambda x: x['confidence'], reverse=True)
        return food_objects
        
    def estimate_distance(self, area):
        """Estimate distance based on object area"""
        ref_area = self.distance_calibration['reference_area']
        ref_distance = self.distance_calibration['reference_distance']
        
        if area > 0:
            distance = ref_distance * np.sqrt(ref_area / area)
            return np.clip(distance, 0.1, 5.0)
        return 5.0
        
    def calculate_angle(self, center_x, image_width):
        """Calculate angle of food relative to robot center"""
        image_center = image_width / 2
        pixel_offset = center_x - image_center
        angle_per_pixel = self.camera_fov / image_width
        angle = pixel_offset * angle_per_pixel
        return np.clip(angle, -30, 30)
        
    def calculate_confidence(self, contour, area, solidity):
        """Calculate detection confidence"""
        # Base confidence from solidity
        confidence = solidity
        
        # Boost confidence for good size
        size_factor = min(1.0, area / 500.0)  # Ideal size around 500 pixels
        confidence *= (0.7 + 0.3 * size_factor)
        
        # Boost confidence for good shape
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h
        if 0.7 <= aspect_ratio <= 1.4:  # Square-ish objects
            confidence *= 1.1
            
        return min(1.0, confidence)


def green_food_collection_task2(rob: IRobobo, agent_type: str = 'dqn', mode: str = 'train', 
                               num_episodes: int = 100):
    """
    Task 2: Green Food Collection using OpenCV + Reinforcement Learning
    
    Objective: Collect all 7 green food boxes within 3 minutes using computer vision
    and reinforcement learning.
    
    Args:
        rob: Robot interface (SimulationRobobo or HardwareRobobo)
        agent_type: Type of RL agent ('qlearning', 'dqn', 'policy_gradient', 'actor_critic')
        mode: Training mode ('train', 'evaluate', 'train_and_evaluate')
        num_episodes: Number of episodes to run
        
    Returns:
        Dictionary with training results and metrics
    """
    
    print(f"🎯 Starting Task 2: Green Food Collection with {agent_type.upper()}")
    print("="*50)
    
    # Initialize components
    env_type = "simulation" if isinstance(rob, SimulationRobobo) else "hardware"
    vision_processor = FoodVisionProcessor(environment_type=env_type)
    environment = RobotEnvironment(rob, vision_processor)
    
    # Initialize DQN agent for Task 2 using factory
    agent = create_rl_agent('dqn', state_size=13, action_size=9)
    
    # Training metrics
    metrics = {
        'episode_rewards': [],
        'episode_food_collected': [],
        'episode_times': [],
        'episode_steps': [],
        'success_rate': [],
        'average_collection_time': []
    }
    
    successful_episodes = 0
    success_rate = 0.0
    action_names = environment.action_descriptions
    
    try:
        for episode in range(num_episodes):
            state = environment.reset()
            episode_reward = 0.0
            episode_steps = 0
            start_time = time.time()
            
            print(f"\n📍 Episode {episode + 1}/{num_episodes}")
            
            while True:
                # Choose action using DQN agent
                action = agent.get_action(state)
                
                # Execute action
                next_state, reward, done, info = environment.step(action)
                
                # Store experience and update agent
                if mode in ['train', 'train_and_evaluate']:
                    agent.update(state, action, reward, next_state, done)
                
                # Update metrics
                episode_reward += reward
                episode_steps += 1
                
                # Print progress
                if episode_steps % 10 == 0:
                    food_status = f"Food: {environment.food_collected}/7"
                    time_elapsed = time.time() - start_time
                    time_status = f"Time: {time_elapsed:.1f}s"
                    reward_status = f"Reward: {episode_reward:.1f}"
                    action_status = f"Action: {action_names[action]}"
                    
                    print(f"\r  Step {episode_steps:3d} | {food_status} | {time_status} | {reward_status} | {action_status}", 
                          end='', flush=True)
                
                # Check if food was collected
                if info.get('food_collected', False):
                    print(f"\n  🟢 Food collected! Total: {environment.food_collected}/7")
                
                state = next_state
                
                if done:
                    break
                    
            # Episode completed
            episode_time = time.time() - start_time
            
            # Update metrics
            metrics['episode_rewards'].append(episode_reward)
            metrics['episode_food_collected'].append(environment.food_collected)
            metrics['episode_times'].append(episode_time)
            metrics['episode_steps'].append(episode_steps)
            
            # Success rate
            if environment.food_collected >= 7:
                successful_episodes += 1
            
            success_rate = successful_episodes / (episode + 1)
            metrics['success_rate'].append(success_rate)
            
            # Average collection time per food
            if environment.food_collected > 0:
                avg_time_per_food = episode_time / environment.food_collected
                metrics['average_collection_time'].append(avg_time_per_food)
            else:
                metrics['average_collection_time'].append(180.0)
            
            # Episode summary
            print(f"\n  ✅ Episode {episode + 1} Complete:")
            print(f"     Food Collected: {environment.food_collected}/7")
            print(f"     Episode Time: {episode_time:.1f}s")
            print(f"     Episode Reward: {episode_reward:.1f}")
            print(f"     Success Rate: {success_rate:.2%}")
            
            # Training - DQN training happens during steps via update() method
            # No additional episode-end training needed for DQN
                
        # Training completed
        print(f"\n🎉 Task 2 Training Complete!")
        print(f"Final Success Rate: {success_rate:.2%}")
        print(f"Average Food Collected: {np.mean(metrics['episode_food_collected']):.1f}/7")
        
        # Save model and results
        if mode in ['train', 'train_and_evaluate']:
            model_path = FIGURES_DIR / f'task2_dqn_model_{int(time.time())}.pth'
            agent.save_model(model_path)
            print(f"Model saved to: {model_path}")
            
            # Save metrics
            metrics_path = FIGURES_DIR / f'task2_metrics_{int(time.time())}.json'
            with open(metrics_path, 'w') as f:
                # Convert numpy arrays to lists for JSON serialization
                json_metrics = {}
                for key, value in metrics.items():
                    if isinstance(value, list):
                        json_metrics[key] = [float(x) for x in value]
                    else:
                        json_metrics[key] = float(value)
                json.dump(json_metrics, f, indent=2)
            print(f"Metrics saved to: {metrics_path}")
            
            # Create training plots
            plot_task2_training_progress(metrics)
            
        return metrics
        
    except KeyboardInterrupt:
        print(f"\n⚠️  Training interrupted by user")
        return metrics
    finally:
        # Ensure robot stops
        rob.move_blocking(0, 0, 100)
        if isinstance(rob, SimulationRobobo):
            rob.stop_simulation()


def plot_task2_training_progress(metrics):
    """Create comprehensive plots for Task 2 training progress"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    episodes = range(1, len(metrics['episode_rewards']) + 1)
    
    # Plot 1: Episode Rewards
    axes[0, 0].plot(episodes, metrics['episode_rewards'])
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].grid(True)
    
    # Plot 2: Food Collection Progress
    axes[0, 1].plot(episodes, metrics['episode_food_collected'], 'g-', linewidth=2)
    axes[0, 1].axhline(y=7, color='r', linestyle='--', label='Target (7 foods)')
    axes[0, 1].set_title('Food Collection Progress')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Foods Collected')
    axes[0, 1].set_ylim(0, 8)
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Plot 3: Success Rate
    axes[0, 2].plot(episodes, [x * 100 for x in metrics['success_rate']], 'purple')
    axes[0, 2].set_title('Success Rate (7/7 Foods)')
    axes[0, 2].set_xlabel('Episode')
    axes[0, 2].set_ylabel('Success Rate (%)')
    axes[0, 2].set_ylim(0, 100)
    axes[0, 2].grid(True)
    
    # Plot 4: Episode Duration
    axes[1, 0].plot(episodes, metrics['episode_times'], 'orange')
    axes[1, 0].axhline(y=180, color='r', linestyle='--', label='Time Limit (180s)')
    axes[1, 0].set_title('Episode Duration')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Time (seconds)')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Plot 5: Average Collection Time per Food
    axes[1, 1].plot(episodes, metrics['average_collection_time'], 'brown')
    axes[1, 1].axhline(y=25.7, color='g', linestyle='--', label='Target (25.7s per food)')
    axes[1, 1].set_title('Efficiency: Time per Food')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Seconds per Food')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    # Plot 6: Running Average Reward (last 10 episodes)
    if len(metrics['episode_rewards']) >= 10:
        window_size = min(10, len(metrics['episode_rewards']))
        running_avg = []
        for i in range(window_size - 1, len(metrics['episode_rewards'])):
            avg = np.mean(metrics['episode_rewards'][i - window_size + 1:i + 1])
            running_avg.append(avg)
        
        axes[1, 2].plot(range(window_size - 1, len(metrics['episode_rewards'])), running_avg, 'red')
        axes[1, 2].set_title(f'Running Average Reward (window={window_size})')
        axes[1, 2].set_xlabel('Episode')
        axes[1, 2].set_ylabel('Average Reward')
        axes[1, 2].grid(True)
    
    plt.suptitle('Task 2: Green Food Collection - Training Progress', fontsize=16)
    plt.tight_layout()
    
    timestamp = int(time.time())
    plot_path = FIGURES_DIR / f'task2_training_progress_{timestamp}.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Training progress plot saved to: {plot_path}")


def test_food_vision_system(rob: IRobobo):
    """Test the computer vision system for food detection"""
    
    print("🔍 Testing Food Vision System")
    print("="*50)
    
    env_type = "simulation" if isinstance(rob, SimulationRobobo) else "hardware"
    vision_processor = FoodVisionProcessor(environment_type=env_type)
    
    print(f"Environment: {env_type}")
    print(f"Configuration: {vision_processor.green_ranges}")
    
    if isinstance(rob, SimulationRobobo):
        rob.play_simulation()
    
    for i in range(10):
        print(f"\nFrame {i+1}/10:")
        
        # Get camera image
        camera_frame = rob.read_image_front()
        if camera_frame is None:
            print("  No camera frame received")
            continue
            
        # Detect food
        food_objects, debug_mask = vision_processor.detect_green_food(camera_frame)
        
        print(f"  Detected {len(food_objects)} food objects")
        
        for j, food in enumerate(food_objects):
            print(f"    Food {j+1}: Distance={food['distance']:.2f}m, "
                  f"Angle={food['angle']:.1f}°, Confidence={food['confidence']:.2f}")
        
        # Save debug images
        debug_image_path = FIGURES_DIR / f'food_detection_frame_{i+1}.png'
        cv2.imwrite(str(debug_image_path), camera_frame)
        
        mask_image_path = FIGURES_DIR / f'food_mask_frame_{i+1}.png'
        cv2.imwrite(str(mask_image_path), debug_mask)
        
        time.sleep(1)
    
    print(f"\n✅ Vision test complete. Debug images saved to {FIGURES_DIR}")
    
    if isinstance(rob, SimulationRobobo):
        rob.stop_simulation()


# ============================================================================
# MAIN ENTRY POINT AND TESTING FUNCTIONS
# ============================================================================

def demo_task2_food_collection(rob: IRobobo):
    """
    Demo function to show Task 2 capabilities
    """
    print("🎯 Task 2 Demo: Green Food Collection")
    print("="*50)
    
    # Test vision system first
    test_food_vision_system(rob)
    
    print("\n🚀 Starting food collection training...")
    
    # Run a short training session
    results = green_food_collection_task2(
        rob=rob,
        agent_type='dqn',
        mode='train_and_evaluate',
        num_episodes=5  # Short demo
    )
    
    print(f"\n📊 Demo Results:")
    if results['episode_food_collected']:
        avg_food = np.mean(results['episode_food_collected'])
        max_food = max(results['episode_food_collected'])
        print(f"  Average Food Collected: {avg_food:.1f}/7")
        print(f"  Best Episode: {max_food}/7 foods")
        
    if results['success_rate']:
        final_success_rate = results['success_rate'][-1]
        print(f"  Final Success Rate: {final_success_rate:.1%}")
    
    return results


def test_task2_capabilities(rob: IRobobo):
    """
    Test Task 2 capabilities and system components
    """
    print("🧪 Task 2 Capability Testing")
    print("="*50)
    
    print("Task 2 - Green Food Collection:")
    print("  Objective: Collect 7 green food boxes in 3 minutes")
    print("  Sensors: IR sensors + camera (OpenCV) + orientation")
    print("  Reward: Food collection + efficiency + approach progress")
    print("  State Space: 13D (8 IR + 3 vision + 2 orientation)")
    print("  Actions: 9 discrete actions optimized for collection")
    
    print("\nKey Features:")
    print("  🎯 Goal: Active food collection with time pressure")
    print("  👁️  Vision: OpenCV green object detection")
    print("  ⏱️  Time: 180 second episode limit")
    print("  🎁 Reward: Collection + approach + efficiency bonuses")
    print("  🧠 Strategy: Target-seeking with obstacle avoidance")
    
    print("\n🔧 Testing Vision System...")
    test_food_vision_system(rob)
    
    print(f"\n🔄 Running Task 2 Demo (2 episodes)...")
    green_food_collection_task2(rob, mode='evaluate', num_episodes=2)


if __name__ == "__main__":
    print("🤖 Robobo Learning Machines - Task 2 Implementation")
    print("="*60)
    print("This file implements Task 2: Green Food Collection")
    print("Key Features:")
    print("  🎯 OpenCV-based green food detection")
    print("  🧠 DQN agent for intelligent navigation")
    print("  📊 Dual masking for simulation/hardware")
    print("  ⏱️  Time-critical 3-minute collection task")
    print("  📈 Comprehensive training and evaluation")
    print("\nMain Functions:")
    print("  • run_green_food_collection(robot) - Primary entry point")
    print("  • green_food_collection_task2(robot) - Core training function")
    print("  • test_food_vision_system(robot) - Vision system testing")
    print("  • demo_task2_food_collection(robot) - Quick demo")
    print("\nExample Usage:")
    print("  from test_actions import run_green_food_collection")
    print("  results = run_green_food_collection(robot, mode='train', rl_episodes=100)")
