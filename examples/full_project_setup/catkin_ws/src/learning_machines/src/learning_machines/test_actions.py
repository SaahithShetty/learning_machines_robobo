"""
Task 2: Green Food Collection - Reinforcement Learning Implementation
====================================================================

RECENT CHANGES (Anti-Wall-Collision Update):
- Removed Stop action from action space (8 actions instead of 9)
- Removed continuous forward momentum rewards that encouraged wall-hitting
- Added stronger collision penalties (25+ points) with escalation
- Added penalties for repetitive action patterns (backward/turn spam)
- Forward movement only rewarded when food is detected or safe exploration
- Action indices updated throughout codebase

LATEST CHANGES (CV_DQN.py Inspired Simplification):
- Simplified reward function based on CV_DQN.py approach
- Removed complex vision-based food detection logic
- Food collection now uses simulation's get_nr_food_collected() method
- Simplified collision detection using basic IR sensor thresholds
- Focus on: food collection (+100), forward movement (+1), green pixel maximization

This module implements Task 2 for the Robobo robot using reinforcement learning
and computer vision. The robot must collect 7 green food boxes within 3 minutes
using DQN (Deep Q-Network) for intelligent navigation and OpenCV for food detection.

Key Components:
- RobotEnvironment: RL environment wrapper optimized for food collection
- DQNAgent: Deep Q-Network agent for action selection and learning
- FoodVisionProcessor: Computer vision system for green food detection
- Comprehensive reward system balancing collection, exploration, and safety

State Space (11D): [8 IR sensors + 3 vision features]
Action Space (8D): backward, turns (left/right), forward movements
Sensor Order: [BackL, BackR, FrontL, FrontR, FrontC, FrontRR, BackC, FrontLL]

Author: Learning Machines Team
Focus: Task 2 (Green Food Collection) - Obstacle Avoidance (Task 1) removed
"""

import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
from datetime import datetime
import json
import math
import pandas as pd
from pathlib import Path
import random
import pickle
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

from typing import Optional, Tuple

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
    action_names = ["Back", "TurnL", "TurnL_S", "FwdL", "Forward", "FwdR", "TurnR_S", "TurnR"]
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
        f"FoodCol:{info.get('food_collision', False)} | "
        f"ObsCol:{info.get('obstacle_collision', False)}"
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
    # Pan/tilt control removed - robot now uses default camera orientation
    print("Phone pan (read-only): ", rob.read_phone_pan())
    print("Phone tilt (read-only): ", rob.read_phone_tilt())


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


def run_all_actions(rob: IRobobo, rl_agent_type: str = 'dqn', 
                   rl_mode: str = 'train', rl_episodes: int = 100, 
                   collision_threshold: float = 0.95):
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
    
    rob.set_phone_tilt_blocking(100, 100)
    rob.set_phone_pan_blocking(180, 50) # Center camera for optimal food detection
    print("="*50)
    print(f"TASK 2: GREEN FOOD COLLECTION USING OPENCV + RL ({rl_agent_type.upper()})")
    print("="*50)
    
    # Run Task 2 - Green Food Collection
    results = green_food_collection_task2(
        rob, agent_type=rl_agent_type, mode=rl_mode, num_episodes=rl_episodes,
        collision_threshold=collision_threshold
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
    - Providing a 11-dimensional state space: [8 IR sensors + 3 vision]
    - Supporting 9 discrete actions optimized for food collection
    """
    
    def __init__(self, robot: IRobobo, vision_processor, max_episode_time: int = 180, 
                 collision_threshold: float = 0.95):
        self.robot = robot
        self.vision_processor = vision_processor
        self.max_episode_time = max_episode_time  # 3 minutes for Task 2
        self.step_count = 0
        
        # Detect environment type for proper camera positioning
        self.is_simulation = hasattr(self.robot, '_smartphone_camera')
        env_type = "simulation" if self.is_simulation else "hardware"
        print(f"Initializing robot environment for {env_type}")
        
        # Unified collision threshold (same for simulation and hardware)
        self.collision_threshold = collision_threshold
        
        # Action space: Task 2 specific actions optimized for food collection
        self.action_space_size = 8
        self.actions = [
            (-50, -50),  # 0: Backward
            (-30, 30),   # 1: Turn Left
            (-15, 30),   # 2: Turn Left Slight
            (40, 60),    # 3: Forward Left
            (60, 60),    # 4: Forward
            (60, 40),    # 5: Forward Right
            (30, -15),   # 6: Turn Right Slight
            (30, -30),   # 7: Turn Right
        ]
        
        # Action descriptions for debugging
        self.action_descriptions = [
            "Backward", "Turn Left", "Turn Left Slight", "Forward Left", 
            "Forward", "Forward Right", "Turn Right Slight", "Turn Right"
        ]
        
        # State space: IR sensors (8) + vision data (3) = 11 dimensions
        # Vision data: [food_detected, food_distance, food_angle]
        self.state_size = 11
        
        # Task 2 specific tracking
        self.food_collected = 0
        self.total_food_target = 7
        self.episode_start_time = None
        self.last_food_positions = []
        
        # Collision state tracking (prevent duplicate collection)
        self.last_collision_step = -1
        self.collision_cooldown = 3  # Prevent multiple collections within N steps
        
        # Action history for temporal learning and grace periods
        self.action_history = deque(maxlen=10)  # Track last 10 actions
        self.reward_history = deque(maxlen=10)  # Track last 10 rewards
        self.grace_period_steps = 3  # Steps to wait before penalizing certain actions
        
        # Wall collision pattern detection
        self.wall_collision_history = deque(maxlen=5)  # Track recent wall collisions
        self.last_wall_collision_step = -1
        self.repeated_collision_penalty = 0  # Escalating penalty for repeated collisions
        
        # Strategy detection for intelligent penalization
        self.consecutive_same_actions = 0
        self.last_action = -1
        self.exploration_bonus_cooldown = 0  # Cooldown for exploration bonuses
        
        # Panoramic scan optimization
        self.scan_frequency = 3  # Scan every N steps to balance speed vs coverage
        self.last_scan_step = 0
        self.cached_vision_data = {'food_detected': 0.0, 'food_distance': 1.0, 'food_angle': 0.0}
        
        # Remove threshold-based obstacle detection - use continuous proximity instead
        # self.obstacle_threshold removed - now using smooth proximity penalties
        
        self.reset()
    
    def _initialize_camera(self):
        """Initialize camera - using default position (no pan/tilt changes)"""
        try:
            # Camera starts in default forward-facing position
            # No pan/tilt adjustments needed - robot starts with camera facing forward
            print(f"Camera using default position - Pan: {self.robot.read_phone_pan()}, Tilt: {self.robot.read_phone_tilt()}")
        except Exception as e:
            print(f"Warning: Could not read camera position: {e}")

    def reset(self):
        """Reset environment for new episode"""
        self.step_count = 0
        
        if self.is_simulation:
            # Reset robot position in simulation
            self.robot.stop_simulation()
            time.sleep(0.5)
            self.robot.play_simulation()
            time.sleep(1.0)
        
        # Initialize camera for optimal food detection
        self._initialize_camera()
        
        # Reset Task 2 specific counters
        self.food_collected = 0
        
        # Use simulation time for simulator, real time for hardware
        if hasattr(self.robot, 'get_sim_time'):
            # Simulation robot - use simulation time for proper headless speedup
            self.episode_start_time = self.robot.get_sim_time()
        else:
            # Hardware robot - use real time
            self.episode_start_time = time.time()
        
        # Reset collision state tracking
        self.last_collision_step = -1
        
        # Sync with simulation's actual food count if available
        if hasattr(self.robot, 'get_nr_food_collected'):
            try:
                sim_food_count = self.robot.get_nr_food_collected()
                print(f"📊 Simulation food count at reset: {sim_food_count}")
                self.food_collected = sim_food_count
            except Exception as e:
                print(f"⚠️  Could not get simulation food count: {e}")
        
        # Debug: Check initial state and camera setup
        self._debug_camera_position()
        initial_state = self._get_state()
        ir_raw = self.robot.read_irs()
        print(f"Reset - IR readings: {ir_raw[:4]}, Food collected: {self.food_collected}")
        
        return initial_state
    
    def _debug_camera_position(self):
        """Debug camera positioning and food detection"""
        try:
            print(f"\n🔍 Camera Debug Info:")
            print(f"  Current Pan: {self.robot.read_phone_pan()}")
            print(f"  Current Tilt: {self.robot.read_phone_tilt()}")
            
            # Test if camera is working
            camera_frame = self.robot.read_image_front()
            if camera_frame is not None:
                print(f"  Camera frame size: {camera_frame.shape}")
                
                # Test food detection
                food_objects, _ = self.vision_processor.detect_green_food(camera_frame)
                print(f"  Food objects detected: {len(food_objects)}")
                
                for i, food in enumerate(food_objects[:3]):  # Show first 3
                    print(f"    Food {i+1}: Distance={food['distance']:.3f}, Angle={food['angle']:.1f}°")
            else:
                print("  ❌ No camera frame received!")
                
        except Exception as e:
            print(f"  Camera debug error: {e}")

    def _sync_food_count_with_simulation(self):
        """Sync our food counter with the simulation's ground truth"""
        if hasattr(self.robot, 'get_nr_food_collected'):
            try:
                sim_food_count = self.robot.get_nr_food_collected()
                if sim_food_count != self.food_collected:
                    print(f"🔄 Food count sync: Python={self.food_collected} → Simulation={sim_food_count}")
                    old_count = self.food_collected
                    self.food_collected = sim_food_count
                    return sim_food_count - old_count  # Return the difference
                return 0
            except Exception as e:
                print(f"⚠️  Could not sync food count with simulation: {e}")
                return 0
        return 0

    def _get_state(self):
        """Get current state representation for Task 2: [IR sensors + vision]"""
        # IR sensors (8 dimensions) - Correct order: [BackL, BackR, FrontL, FrontR, FrontC, FrontRR, BackC, FrontLL]
        ir_values = self.robot.read_irs()
        ir_normalized = []
        for val in ir_values:
            if val is None:
                ir_normalized.append(0.0)  # No detection
            else:
                # Unified IR sensor normalization (eliminate reality gap)
                # Normalize all sensor values to 0-1 range for consistent behavior
                ir_normalized.append(min(val / 1000.0, 1.0))
        
        # Vision data (3 dimensions: food_detected, food_distance, food_angle)
        # Use panoramic detection for wider field of view (~75° total coverage)
        # Scans left (-25°), center (0°), right (+25°) to detect food at wider angles
        vision_data = self._get_panoramic_food_state()
        food_detected = vision_data[0]  # Binary: food visible
        food_distance = vision_data[1]  # Normalized distance [0,1]
        food_angle = vision_data[2]     # Normalized angle [-1,1]
        
        # Combine: 8 IR + 3 vision = 11 dimensions
        state = np.array(ir_normalized + [food_detected, food_distance, food_angle], 
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
        miniduration = 300
        # if left_speed == right_speed  and left_speed > 0:
        #     miniduration = 300
        self.robot.move(left_speed, right_speed, miniduration)

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
        elif info.get('obstacle_collision', False):
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
        info['action_taken'] = self.action_descriptions[action_idx]
        info['food_collected_count'] = int(self.food_collected)
        
        # Check if episode is done - Task 2: Only time limit and food collection target
        # Use appropriate time measurement based on robot type
        if hasattr(self.robot, 'get_sim_time'):
            # Simulation robot - use simulation time for proper headless speedup
            time_elapsed = self.robot.get_sim_time() - self.episode_start_time if self.episode_start_time else 0
        else:
            # Hardware robot - use real time
            time_elapsed = time.time() - self.episode_start_time if self.episode_start_time else 0
            
        done = (
            time_elapsed >= self.max_episode_time or
            self.food_collected >= self.total_food_target
        )
        
        # Add episode completion reward/penalty
        if done:
            if self.food_collected >= self.total_food_target:
                # Success: All foods collected
                time_remaining = max(0, self.max_episode_time - time_elapsed)
                success_bonus = 500 + (time_remaining / self.max_episode_time) * 200  # Big bonus for success + speed
                reward += success_bonus
                print(f"🏆 EPISODE SUCCESS! All {self.total_food_target} foods collected! Bonus: {success_bonus:.1f}")
            elif time_elapsed >= self.max_episode_time:
                # Failure: Time ran out
                failure_penalty = -200  # Penalty for not completing in time
                reward += failure_penalty
                print(f"⏰ EPISODE TIMEOUT! Only {self.food_collected}/{self.total_food_target} foods collected. Penalty: {failure_penalty}")
            
            info['episode_complete'] = True
            info['success'] = self.food_collected >= self.total_food_target
        
        # Sync with simulation food count for accurate tracking
        food_collected_this_step = self._sync_food_count_with_simulation()
        if food_collected_this_step > 0:
            info['food_collected'] = True
            print(f"🎉 FOOD COLLECTED! (Simulation confirmed +{food_collected_this_step})")
        
        # Note: The reward calculation in _calculate_reward already handles 
        # food collection detection and sets info['food_collected']
        
        return next_state, reward, done, info
    
    def _calculate_reward(self, action_idx: int, state: np.ndarray) -> Tuple[float, dict]:
        """ENHANCED FOOD-CENTRIC REWARD FUNCTION
        
        IMPROVED REWARD HIERARCHY:
        ==========================
        1. LINEAR FOOD COLLECTION: +100 * (1 + food_collected/10) 
        2. SAFETY COLLISION: -10 (obstacle collision penalty)
        3. NAVIGATION GUIDANCE: +0.1 to +10.0 max per step
        4. DISTANCE-BASED FOOD REWARDS: Approach guidance
        5. AREA COVERAGE: Exploration rewards
        
        This provides stronger guidance while maintaining food-centricity.
        """
        reward = 0.0
        info = {}
        
        # Extract state components
        ir_values = list(state[:8])
        
        # =================================================================
        # 1. LINEAR FOOD COLLECTION - Major Reward Source
        # =================================================================
        
        food_collision_detected = self._check_food_collision()
        if food_collision_detected:
            # Linear increase: first food = 100, second = 110, third = 120, etc.
            linear_bonus = 100 * (1 + self.food_collected * 0.1)
            reward += linear_bonus
            info['food_collected'] = True
            info['linear_food_reward'] = linear_bonus
            print(f"🎉 FOOD COLLECTED! Linear Reward: +{linear_bonus:.1f} (Food #{self.food_collected})")
            return reward, info  # Early return - nothing else matters when food is collected
        
        # =================================================================
        # 2. SAFETY - Obstacle Collision Penalty
        # =================================================================
        
        min_distance = min(ir_values) if ir_values else 1.0
        if min_distance < 0.1:  # Very close to something
            # Check if this is an obstacle collision (not food)
            camera_frame = self.robot.read_image_front()
            is_food_collision = self._is_colliding_with_food(camera_frame)
            
            if not is_food_collision:
                # Safety collision penalty
                reward -= 10  
                info['obstacle_collision'] = True
                info['collision_penalty'] = -10
                info['collision_type'] = 'obstacle'
            else:
                info['collision_type'] = 'food'  # No penalty for food collisions
        
        # =================================================================
        # 3. ENHANCED NAVIGATION GUIDANCE (Max +10.0 per step)
        # =================================================================
        
        # Get camera frame for vision guidance
        camera_frame = self.robot.read_image_front()
        if camera_frame is not None:
            left_green, middle_green, right_green = self.vision_processor._analyze_green_sections(camera_frame)
            
            # Enhanced reward for seeing food (up to 4.0)
            total_green = left_green + middle_green + right_green
            green_bonus = min(total_green * 0.15, 4.0)  # Increased scale and cap
            reward += green_bonus
            info['green_sight_bonus'] = green_bonus
            
            # Enhanced reward for centering food (up to 3.0)
            if middle_green > max(left_green, right_green) and middle_green > 1.0:
                center_bonus = min(middle_green * 0.15, 3.0)  # Scale with amount of food in center
                reward += center_bonus
                info['center_bonus'] = center_bonus
            
            # Direction guidance bonus (up to 2.0)
            if left_green > right_green + 2.0:  # Food more on left
                if action_idx in [1, 2, 3]:  # Left turning/forward-left actions
                    direction_bonus = min(left_green * 0.1, 1.5)
                    reward += direction_bonus
                    info['direction_guidance'] = direction_bonus
            elif right_green > left_green + 2.0:  # Food more on right  
                if action_idx in [5, 6, 7]:  # Right turning/forward-right actions
                    direction_bonus = min(right_green * 0.1, 1.5)
                    reward += direction_bonus
                    info['direction_guidance'] = direction_bonus
        
        # Enhanced forward movement reward (up to 1.0)
        if action_idx == 4 and camera_frame is not None:  # Forward action
            left_green, middle_green, right_green = self.vision_processor._analyze_green_sections(camera_frame)
            if (left_green + middle_green + right_green) > 1.0:  # Only reward forward if food is visible
                forward_bonus = min((left_green + middle_green + right_green) * 0.08, 1.0)
                reward += forward_bonus
                info['forward_toward_food'] = forward_bonus
        
        # =================================================================
        # 4. DISTANCE-BASED FOOD DETECTION AND REWARD
        # =================================================================
        
        distance_reward = self._calculate_distance_based_food_reward(camera_frame)
        if distance_reward > 0:
            reward += distance_reward
            info['distance_food_reward'] = distance_reward
        
        # =================================================================
        # 5. AREA COVERAGE REWARDS FOR EXPLORATION
        # =================================================================
        
        exploration_reward = self._calculate_exploration_reward()
        if exploration_reward > 0:
            reward += exploration_reward
            info['exploration_reward'] = exploration_reward
        
        # =================================================================
        # 6. PREVENT DESTRUCTIVE BEHAVIORS (Minimal penalties)
        # =================================================================
        
        # Track repetitive actions (prevent getting stuck)
        if not hasattr(self, 'action_history'):
            self.action_history = deque(maxlen=8)  # Longer history for better detection
        self.action_history.append(action_idx)
        
        if len(self.action_history) >= 6:
            recent_actions = list(self.action_history)[-6:]
            current_action_count = recent_actions.count(action_idx)
            
            # Penalize excessive repetition (5+ times in last 6 actions) - more lenient
            if current_action_count >= 5:
                if action_idx == 0:  # Backward spam
                    reward -= 5  # Reduced penalty but still significant
                    info['backward_spam_penalty'] = -5
                    print("🔄 BACKWARD SPAM: -5")
                elif action_idx in [1, 2, 6, 7]:  # Turn spam
                    reward -= 3  # Reduced penalty
                    info['turn_spam_penalty'] = -3
                    print("🔄 TURN SPAM: -3")
        
        # Store state info
        info['ir_sensors'] = ir_values
        info['action_taken'] = action_idx
        info['min_distance'] = min_distance
        
        return reward, info
        
    def _calculate_distance_based_food_reward(self, camera_frame) -> float:
        """Calculate reward based on distance to detected food
        
        Returns higher rewards for getting closer to food objects.
        This provides navigation guidance toward food.
        """
        if camera_frame is None:
            return 0.0
        
        try:
            # Detect food objects with distance information
            food_objects, _ = self.vision_processor.detect_green_food(camera_frame)
            
            if not food_objects:
                return 0.0
            
            # Find closest food object
            closest_food = min(food_objects, key=lambda f: f['distance'])
            distance = closest_food['distance']
            angle = abs(closest_food['angle'])
            
            # Enhanced distance-based reward: closer = higher reward
            # Scale: distance 0.0-1.0 maps to reward 3.0-0.0 (increased from 2.0)
            distance_reward = max(0.0, 3.0 * (1.0 - distance))
            
            # Angle bonus: facing food directly gives extra reward
            if angle < 10:  # Within 10 degrees (more strict)
                distance_reward *= 2.0  # 100% bonus for excellent alignment
            elif angle < 20:  # Within 20 degrees  
                distance_reward *= 1.5  # 50% bonus for good alignment
            elif angle < 40:  # Within 40 degrees
                distance_reward *= 1.2  # 20% bonus for reasonable alignment
            
            # Progressive approach bonus - extra reward for very close food
            if distance < 0.3:  # Very close
                proximity_bonus = (0.3 - distance) * 5.0  # Up to 1.5 bonus
                distance_reward += proximity_bonus
            
            return min(distance_reward, 5.0)  # Cap at 5.0 (increased from 3.0)
            
        except Exception as e:
            print(f"Error calculating distance-based food reward: {e}")
            return 0.0
    
    def _calculate_exploration_reward(self) -> float:
        """Calculate reward for area coverage and exploration
        
        Encourages the robot to explore different areas rather than
        getting stuck in one location. Tracks visited positions.
        """
        try:
            # Initialize exploration tracking if needed
            if not hasattr(self, 'visited_positions'):
                self.visited_positions = set()
                self.last_position = None
                self.exploration_grid_size = 0.15  # Smaller grid for finer coverage tracking
                self.steps_since_exploration = 0
            
            # Get current position (estimated from IR sensors and movement)
            current_pos = self._estimate_current_position()
            
            if current_pos is None:
                return 0.0
            
            # Discretize position to grid
            grid_x = int(current_pos[0] / self.exploration_grid_size)
            grid_y = int(current_pos[1] / self.exploration_grid_size)
            grid_pos = (grid_x, grid_y)
            
            # Check if this is a new area
            if grid_pos not in self.visited_positions:
                self.visited_positions.add(grid_pos)
                self.steps_since_exploration = 0
                
                # Enhanced reward for discovering new area
                exploration_reward = 2.0  # Increased from 1.0
                
                # Progressive bonus for covering more area overall
                coverage_bonus = min(len(self.visited_positions) * 0.15, 2.0)  # Increased scale
                total_reward = exploration_reward + coverage_bonus
                
                return min(total_reward, 3.0)  # Cap at 3.0 (increased from 2.0)
            else:
                # Track time since last exploration
                self.steps_since_exploration += 1
                
                # Small penalty for staying in same areas too long
                if self.steps_since_exploration > 10:
                    stagnation_penalty = min((self.steps_since_exploration - 10) * 0.1, 1.0)
                    return -stagnation_penalty
            
            # Small reward for movement (prevents staying in one spot)
            if self.last_position is not None:
                movement_distance = np.linalg.norm(np.array(current_pos) - np.array(self.last_position))
                movement_reward = min(movement_distance * 1.0, 0.5)  # Increased movement reward
                self.last_position = current_pos
                return movement_reward
            
            self.last_position = current_pos
            return 0.0
            
        except Exception as e:
            print(f"Error calculating exploration reward: {e}")
            return 0.0
    
    def _estimate_current_position(self) -> Optional[Tuple[float, float]]:
        """Estimate current position based on movement history and sensors
        
        This is a simple position estimation for exploration tracking.
        In a real system, you might use odometry or SLAM.
        """
        try:
            # Simple position estimation based on step count and action history
            if not hasattr(self, 'estimated_x'):
                self.estimated_x = 0.0
                self.estimated_y = 0.0
                self.estimated_heading = 0.0  # Radians
            
            # Get the last action if available
            if hasattr(self, 'action_history') and len(self.action_history) > 0:
                last_action = self.action_history[-1]
                
                # Estimate movement based on action
                movement_step = 0.05  # Estimated meters per step
                
                if last_action == 4:  # Forward
                    self.estimated_x += movement_step * np.cos(self.estimated_heading)
                    self.estimated_y += movement_step * np.sin(self.estimated_heading)
                elif last_action == 0:  # Backward
                    self.estimated_x -= movement_step * np.cos(self.estimated_heading)
                    self.estimated_y -= movement_step * np.sin(self.estimated_heading)
                elif last_action in [1, 2]:  # Turn left
                    self.estimated_heading += 0.1  # Radians
                elif last_action in [6, 7]:  # Turn right
                    self.estimated_heading -= 0.1  # Radians
                elif last_action == 3:  # Forward left
                    self.estimated_x += movement_step * 0.7 * np.cos(self.estimated_heading + 0.2)
                    self.estimated_y += movement_step * 0.7 * np.sin(self.estimated_heading + 0.2)
                elif last_action == 5:  # Forward right
                    self.estimated_x += movement_step * 0.7 * np.cos(self.estimated_heading - 0.2)
                    self.estimated_y += movement_step * 0.7 * np.sin(self.estimated_heading - 0.2)
                
                # Keep heading in [-π, π] range
                self.estimated_heading = ((self.estimated_heading + np.pi) % (2 * np.pi)) - np.pi
            
            return (self.estimated_x, self.estimated_y)
            
        except Exception as e:
            print(f"Error estimating position: {e}")
            return None
    
    def _check_food_collision(self):
        """Simple food collection detection like CV_DQN.py
        
        Uses simulation's built-in food collection tracking rather than 
        complex IR+camera collision detection.
        """
        if hasattr(self.robot, 'get_nr_food_collected'):
            current_count = self.robot.get_nr_food_collected()
            if current_count > self.food_collected:
                # Food was collected since last check
                food_collected_this_step = current_count - self.food_collected
                self.food_collected = current_count
                return food_collected_this_step > 0
        return False
    
    def _is_colliding_with_food(self, camera_frame):
        """Check if current collision is with food vs obstacle
        
        Returns True if significant green (food) is visible during collision,
        indicating the robot is touching a food box rather than a wall.
        """
        if camera_frame is None:
            return False
        
        try:
            # Get green pixel distribution
            left_green, middle_green, right_green = self.vision_processor._analyze_green_sections(camera_frame)
            
            # If significant green is visible, especially in center, it's likely food collision
            total_green = left_green + middle_green + right_green
            center_green_dominant = middle_green > max(left_green, right_green)
            
            # Food collision criteria:
            # 1. Substantial green visible (>5% of view)
            # 2. Green concentrated in center (robot facing food)
            is_food_collision = (total_green > 5.0) and (center_green_dominant or middle_green > 8.0)
            
            return is_food_collision
            
        except Exception as e:
            print(f"Error checking food collision: {e}")
            return False
    
    def _get_panoramic_food_state(self):
        """Get enhanced food detection state from camera
        
        Returns comprehensive vision data for the state representation.
        """
        try:
            camera_frame = self.robot.read_image_front()
            if camera_frame is None:
                return [0.0, 1.0, 0.0]  # Default: no food detected, max distance, centered
            
            # Detect food objects
            food_objects, processed_frame = self.vision_processor.detect_green_food(camera_frame)
            
            if not food_objects:
                return [0.0, 1.0, 0.0]  # No food visible
            
            # Find closest food
            closest_food = min(food_objects, key=lambda f: f['distance'])
            
            # Extract features
            food_detected = 1.0  # Binary: food visible
            distance_normalized = min(closest_food['distance'], 1.0)  # Clamp to [0,1]
            angle_normalized = closest_food['angle'] / 90.0  # Normalize angle to [-1,1]
            
            return [food_detected, distance_normalized, angle_normalized]
            
        except Exception as e:
            print(f"Error getting panoramic food state: {e}")
            return [0.0, 1.0, 0.0]  # Default safe values


class FoodVisionProcessor:
    """Computer Vision system for green food detection and localization
    
    This class handles all computer vision tasks for Task 2:
    - Green color detection and masking
    - Food object localization and distance estimation
    - Multi-angle scanning for comprehensive food detection
    - Noise filtering and robust detection algorithms
    """
    
    def __init__(self, debug_mode: bool = False):
        self.debug_mode = debug_mode
        
        # Green color range for food detection (HSV)
        # Optimized for both simulation and hardware lighting conditions
        self.green_lower = np.array([35, 50, 50])   # Lower HSV threshold
        self.green_upper = np.array([85, 255, 255]) # Upper HSV threshold
        
        # Detection parameters
        self.min_contour_area = 100      # Minimum pixels for valid food detection
        self.max_detection_distance = 2.0 # Maximum detection range (meters)
        
        # Noise filtering
        self.gaussian_blur_kernel = (5, 5)
        self.morphology_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        
    def detect_green_food(self, camera_frame):
        """Detect green food objects in camera frame
        
        Returns:
            food_objects: List of detected food with [distance, angle, confidence]
            processed_frame: Debug frame showing detection results
        """
        if camera_frame is None:
            return [], None
            
        try:
            # Convert to HSV for better color detection
            hsv = cv2.cvtColor(camera_frame, cv2.COLOR_BGR2HSV)
            
            # Create mask for green color
            green_mask = cv2.inRange(hsv, self.green_lower, self.green_upper)
            
            # Noise reduction
            green_mask = cv2.GaussianBlur(green_mask, self.gaussian_blur_kernel, 0)
            green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, self.morphology_kernel)
            green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, self.morphology_kernel)
            
            # Find contours
            contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            food_objects = []
            processed_frame = camera_frame.copy() if self.debug_mode else None
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > self.min_contour_area:
                    # Calculate food properties
                    food_info = self._analyze_food_contour(contour, camera_frame.shape)
                    if food_info:
                        food_objects.append(food_info)
                        
                        # Draw debug info
                        if self.debug_mode and processed_frame is not None:
                            cv2.drawContours(processed_frame, [contour], -1, (0, 255, 0), 2)
                            
            return food_objects, processed_frame
            
        except Exception as e:
            print(f"Error in green food detection: {e}")
            return [], None
    
    def _analyze_food_contour(self, contour, frame_shape):
        """Analyze food contour to extract distance and angle information"""
        try:
            # Calculate centroid
            M = cv2.moments(contour)
            if M["m00"] == 0:
                return None
                
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            # Frame dimensions
            height, width = frame_shape[:2]
            
            # Calculate angle (horizontal position relative to center)
            center_x = width // 2
            pixel_offset = cx - center_x
            
            # Convert pixel offset to angle (assuming ~60° field of view)
            angle_degrees = (pixel_offset / center_x) * 30  # ±30° range
            
            # Estimate distance based on contour area (rough approximation)
            area = cv2.contourArea(contour)
            distance = max(0.1, min(2.0, 1000.0 / (area + 1)))  # Inverse relationship
            
            # Calculate confidence based on area and shape
            confidence = min(1.0, area / 1000.0)
            
            return {
                'distance': distance,
                'angle': angle_degrees,
                'confidence': confidence,
                'centroid': (cx, cy),
                'area': area
            }
            
        except Exception as e:
            print(f"Error analyzing food contour: {e}")
            return None
    
    def _analyze_green_sections(self, camera_frame):
        """Analyze green content in left, middle, right sections of frame
        
        Returns: (left_green_percent, middle_green_percent, right_green_percent)
        """
        if camera_frame is None:
            return 0.0, 0.0, 0.0
            
        try:
            # Convert to HSV
            hsv = cv2.cvtColor(camera_frame, cv2.COLOR_BGR2HSV)
            green_mask = cv2.inRange(hsv, self.green_lower, self.green_upper)
            
            # Divide frame into three sections
            height, width = green_mask.shape
            section_width = width // 3
            
            left_section = green_mask[:, :section_width]
            middle_section = green_mask[:, section_width:2*section_width]
            right_section = green_mask[:, 2*section_width:]
            
            # Calculate green percentage in each section
            total_pixels_per_section = height * section_width
            
            left_green = (np.sum(left_section) / 255) / total_pixels_per_section * 100
            middle_green = (np.sum(middle_section) / 255) / total_pixels_per_section * 100
            right_green = (np.sum(right_section) / 255) / total_pixels_per_section * 100
            
            return left_green, middle_green, right_green
            
        except Exception as e:
            print(f"Error analyzing green sections: {e}")
            return 0.0, 0.0, 0.0


def green_food_collection_task2(rob: IRobobo, agent_type: str = 'dqn', mode: str = 'train', 
                               num_episodes: int = 100, collision_threshold: float = 0.95,
                               model_path: Optional[str] = None):
    """Main function for Task 2: Green Food Collection using RL + Computer Vision
    
    This function orchestrates the complete food collection task including:
    - Environment setup with computer vision
    - RL agent initialization and training
    - Performance evaluation and metrics collection
    - Model saving and visualization
    
    Args:
        rob: Robot interface (SimulationRobobo or HardwareRobobo)
        agent_type: RL algorithm ('dqn', 'qlearning', 'policy_gradient', 'actor_critic')
        mode: 'train', 'evaluate', or 'train_and_evaluate'
        num_episodes: Number of episodes to run
        collision_threshold: IR sensor threshold for collision detection
        
    Returns:
        Dict with training results, performance metrics, and saved file paths
    """
    
    print(f"\n🎯 TASK 2: GREEN FOOD COLLECTION")
    print(f"Agent: {agent_type.upper()}, Mode: {mode}, Episodes: {num_episodes}")
    print(f"Environment: {'Simulation' if isinstance(rob, SimulationRobobo) else 'Hardware'}")
    
    # Initialize computer vision processor
    vision_processor = FoodVisionProcessor(debug_mode=True)
    
    # Create environment
    env = RobotEnvironment(rob, vision_processor, max_episode_time=180, 
                          collision_threshold=collision_threshold)
    
    # Get hyperparameters for the chosen agent type
    hyperparams = get_default_hyperparameters(agent_type)
    
    # Create RL agent
    agent = create_rl_agent(
        agent_type=agent_type,
        state_size=env.state_size,
        action_size=env.action_space_size,
        **hyperparams
    )
    
    # Load pre-trained model if provided (for evaluation mode)
    if model_path and mode in ['evaluate', 'test']:
        print(f"Loading pre-trained model: {model_path}")
        try:
            agent.load_model(model_path)
            print("✅ Model loaded successfully!")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            return None
    
    # Performance tracking
    results = {
        'agent_type': agent_type,
        'mode': mode,
        'episodes': num_episodes,
        'episode_rewards': [],
        'episode_foods_collected': [],
        'episode_lengths': [],
        'episode_success_rates': [],
        'training_time': 0,
        'best_episode_reward': float('-inf'),
        'best_foods_collected': 0,
        'model_path': None,
        'metrics_path': None,
        'convergence_episode': None
    }
    
    # Training phase
    if mode in ['train', 'train_and_evaluate']:
        print(f"\n🚀 Starting {agent_type.upper()} training...")
        start_time = time.time()
        
        for episode in range(num_episodes):
            episode_start_time = time.time()
            state = env.reset()
            total_reward = 0
            step_count = 0
            done = False
            
            print(f"\n📍 Episode {episode + 1}/{num_episodes}")
            
            while not done and step_count < 1800:  # 3 minutes at 10 Hz
                # Agent selects action
                if hasattr(agent, 'act'):
                    action = agent.act(state)
                elif hasattr(agent, 'get_action'):
                    action = agent.get_action(state)
                else:
                    # This should not happen with proper agent implementation
                    raise AttributeError(f"Agent {type(agent)} has no 'act' or 'get_action' method")
                
                # Execute action in environment
                next_state, reward, done, info = env.step(action)
                
                # Train agent (if in training mode)
                if mode == 'train' or mode == 'train_and_evaluate':
                    if hasattr(agent, 'learn'):
                        agent.learn(state, action, reward, next_state, done)
                    elif hasattr(agent, 'update'):
                        agent.update(state, action, reward, next_state, done)
                    else:
                        # This should not happen with proper agent implementation
                        raise AttributeError(f"Agent {type(agent)} has no 'learn' or 'update' method")
                
                # Update state and tracking
                state = next_state
                total_reward += reward
                step_count += 1
                
                # Print step info every 50 steps
                if step_count % 50 == 0:
                    foods = info.get('food_collected_count', 0)
                    time_elapsed = time.time() - episode_start_time
                    action_name = env.action_descriptions[action]
                    print(f"  Step {step_count:3d} | Food: {foods}/7 (Sim:{env.food_collected}) | "
                          f"Time: {time_elapsed:.1f}s | Reward: {total_reward:.1f} | "
                          f"Action: {action_name}")
            
            # Episode complete
            episode_time = time.time() - episode_start_time
            foods_collected = env.food_collected
            success = foods_collected >= 7
            
            # Record results
            results['episode_rewards'].append(total_reward)
            results['episode_foods_collected'].append(foods_collected)
            results['episode_lengths'].append(step_count)
            results['episode_success_rates'].append(1.0 if success else 0.0)
            
            # Track best performance
            if total_reward > results['best_episode_reward']:
                results['best_episode_reward'] = total_reward
            if foods_collected > results['best_foods_collected']:
                results['best_foods_collected'] = foods_collected
            
            # Episode summary
            success_rate = np.mean(results['episode_success_rates'][-10:]) * 100  # Last 10 episodes
            print(f"  ✅ Episode {episode + 1} Complete:")
            print(f"     Food Collected: {foods_collected}/7 ({'SUCCESS' if success else 'PARTIAL'})")
            print(f"     Episode Time: {episode_time:.1f}s")
            print(f"     Episode Reward: {total_reward:.1f}")
            print(f"     Success Rate: {success_rate:.1f}%")
            
            # Save model periodically
            if (episode + 1) % 50 == 0:
                timestamp = int(time.time())
                model_filename = f"rl_model_{agent_type}_{timestamp}.pth"
                model_save_path = FIGURES_DIR / model_filename
                agent.save_model(str(model_save_path))
                print(f"     💾 Model saved: {model_filename}")
        
        results['training_time'] = time.time() - start_time
        print(f"\n🎯 Training completed in {results['training_time']:.1f} seconds")
    
    # Evaluation phase
    elif mode in ['evaluate', 'test']:
        print(f"\n🧪 Starting {agent_type.upper()} evaluation...")
        start_time = time.time()
        
        for episode in range(num_episodes):
            episode_start_time = time.time()
            state = env.reset()
            total_reward = 0
            step_count = 0
            done = False
            
            print(f"\n📍 Evaluation Episode {episode + 1}/{num_episodes}")
            
            while not done and step_count < 1800:  # 3 minutes at 10 Hz
                # Agent selects action (no exploration in evaluation mode)
                if hasattr(agent, 'act'):
                    action = agent.act(state, training=False)
                elif hasattr(agent, 'get_action'):
                    action = agent.get_action(state, training=False)
                else:
                    # Fallback for agents without training parameter support
                    action = agent.act(state) if hasattr(agent, 'act') else agent.get_action(state)
                
                # Execute action in environment
                next_state, reward, done, info = env.step(action)
                
                # Update state and tracking (no learning in evaluation mode)
                state = next_state
                total_reward += reward
                step_count += 1
                
                # Print step info every 50 steps
                if step_count % 50 == 0:
                    foods = info.get('food_collected_count', 0)
                    time_elapsed = time.time() - episode_start_time
                    action_name = env.action_descriptions[action]
                    print(f"  Step {step_count:3d} | Food: {foods}/7 (Sim:{env.food_collected}) | "
                          f"Time: {time_elapsed:.1f}s | Reward: {total_reward:.1f} | "
                          f"Action: {action_name}")
            
            # Episode complete
            episode_time = time.time() - episode_start_time
            foods_collected = env.food_collected
            success = foods_collected >= 7
            
            # Record results
            results['episode_rewards'].append(total_reward)
            results['episode_foods_collected'].append(foods_collected)
            results['episode_lengths'].append(step_count)
            results['episode_success_rates'].append(1.0 if success else 0.0)
            
            # Track best performance
            if total_reward > results['best_episode_reward']:
                results['best_episode_reward'] = total_reward
            if foods_collected > results['best_foods_collected']:
                results['best_foods_collected'] = foods_collected
            
            # Episode summary
            success_rate = np.mean(results['episode_success_rates'][-10:]) * 100  # Last 10 episodes
            print(f"  ✅ Evaluation Episode {episode + 1} Complete:")
            print(f"     Food Collected: {foods_collected}/7 ({'SUCCESS' if success else 'PARTIAL'})")
            print(f"     Episode Time: {episode_time:.1f}s")
            print(f"     Episode Reward: {total_reward:.1f}")
            print(f"     Success Rate: {success_rate:.1f}%")
        
        results['evaluation_time'] = time.time() - start_time
        print(f"\n🎯 Evaluation completed in {results['evaluation_time']:.1f} seconds")
    
    # Save final model and results
    timestamp = int(time.time())
    
    # Save model
    if mode in ['train', 'train_and_evaluate']:
        model_filename = f"rl_model_{agent_type}_{timestamp}.pth"
        model_save_path = FIGURES_DIR / model_filename
        agent.save_model(str(model_save_path))
        results['model_path'] = str(model_save_path)
        print(f"✅ Final model saved: {model_filename}")
    
    # Save metrics
    metrics_filename = f"training_metrics_{agent_type}_{timestamp}.json"
    metrics_path = FIGURES_DIR / metrics_filename
    with open(metrics_path, 'w') as f:
        json.dump(results, f, indent=2)
    results['metrics_path'] = str(metrics_path)
    
    # Generate performance plots
    if len(results['episode_rewards']) > 0:
        _plot_training_results(results, agent_type, timestamp)
    
    # Final summary
    if results['episode_rewards']:
        avg_reward = np.mean(results['episode_rewards'])
        avg_foods = np.mean(results['episode_foods_collected'])
        final_success_rate = np.mean(results['episode_success_rates']) * 100
        
        print(f"\n📊 FINAL RESULTS:")
        print(f"   Average Reward: {avg_reward:.1f}")
        print(f"   Average Foods Collected: {avg_foods:.1f}/7")
        print(f"   Success Rate: {final_success_rate:.1f}%")
        print(f"   Best Foods Collected: {results['best_foods_collected']}/7")
    
    return results


def test_task2_capabilities(rob: IRobobo):
    """Test all capabilities required for Task 2: Green Food Collection
    
    This function tests various components needed for Task 2 including:
    - Computer vision for green food detection
    - Movement capabilities for navigation
    - Sensor readings for obstacle avoidance
    - Food collection mechanism
    
    Args:
        rob: Robot interface (SimulationRobobo or HardwareRobobo)
    """
    import time
    import numpy as np
    import cv2
    
    print("\n===== TESTING TASK 2 CAPABILITIES =====")
    
    # Test basic movements
    print("\n1. Testing basic movements...")
    rob.set_emotion(Emotion.NORMAL)
    rob.move(10, 10)  # Slow forward
    time.sleep(1)
    rob.move(0, 0)    # Stop
    time.sleep(0.5)
    rob.move(-10, -10)  # Slow backward
    time.sleep(1)
    rob.move(0, 0)    # Stop
    time.sleep(0.5)
    rob.move(10, -10)  # Turn right
    time.sleep(1)
    rob.move(0, 0)    # Stop
    time.sleep(0.5)
    rob.move(-10, 10)  # Turn left
    time.sleep(1)
    rob.move(0, 0)    # Stop
    
    # Test IR sensors for obstacle detection
    print("\n2. Testing IR sensors...")
    ir_values = rob.read_irs()
    print(f"IR Sensor Values: {ir_values}")
    
    # Test camera for food detection
    print("\n3. Testing camera for food detection...")
    image = rob.read_image_front()
    if image is not None:
        # Simple green detection
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_green = np.array([35, 100, 100])
        upper_green = np.array([85, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)
        green_pixels = np.sum(mask > 0)
        green_percentage = (green_pixels / (image.shape[0] * image.shape[1])) * 100
        
        print(f"Green pixels detected: {green_pixels} ({green_percentage:.2f}%)")
        
        # Save the image and mask for inspection
        if isinstance(rob, SimulationRobobo):
            cv2.imwrite("results/figures/test_camera_rgb.png", image)
            cv2.imwrite("results/figures/test_green_mask.png", mask)
    else:
        print("Failed to capture image from camera")
    
    # Test food collection (simulation only)
    if isinstance(rob, SimulationRobobo):
        print("\n4. Testing food collection (simulation only)...")
        try:
            food_count = rob.get_nr_food_collected()
            print(f"Current food collected: {food_count}")
        except Exception as e:
            print(f"Food collection test failed: {e}")
    
    # Test complete environment integration
    print("\n5. Testing environment integration...")
    try:
        vision_processor = FoodVisionProcessor(debug_mode=True)
        env = RobotEnvironment(rob, vision_processor, max_episode_time=30)
        state = env.reset()
        
        print(f"State shape: {state.shape}")
        print(f"Initial state: {state}")
        
        # Test a few random actions
        for i in range(5):
            action = np.random.randint(0, 8)  # 8 actions (0-7)
            print(f"\nTaking random action: {action}")
            next_state, reward, done, info = env.step(action)
            print(f"Reward: {reward}, Done: {done}")
            print(f"Info: {info}")
            time.sleep(1)
    except Exception as e:
        print(f"Environment integration test failed: {e}")
    
    print("\n===== TASK 2 CAPABILITY TESTING COMPLETE =====")


def demo_task2_food_collection(rob: IRobobo, duration: int = 60):
    """Demo for Task 2: Green Food Collection without RL
    
    This function demonstrates a simple food collection behavior
    without using reinforcement learning. It uses basic computer vision
    and hardcoded behaviors to collect green food items.
    
    Args:
        rob: Robot interface (SimulationRobobo or HardwareRobobo)
        duration: Duration of the demo in seconds
    """
    import time
    import numpy as np
    import cv2
    from datetime import datetime
    
    print("\n===== TASK 2 FOOD COLLECTION DEMO =====")
    print(f"Running demo for {duration} seconds...")
    
    # Initialize vision processor
    vision_processor = FoodVisionProcessor(debug_mode=True)
    
    # Reset robot state
    rob.set_emotion(Emotion.NORMAL)
    rob.move(0, 0)
    
    start_time = datetime.now()
    food_collected = 0
    
    if isinstance(rob, SimulationRobobo):
        food_collected = rob.get_nr_food_collected()
    
    while (datetime.now() - start_time).total_seconds() < duration:
        # Get camera image
        image = rob.read_image_front()
        
        if image is None:
            print("Failed to capture image, skipping frame")
            continue
        
        # Detect green food
        food_info = vision_processor.detect_green_food(image)
        
        # Simple behavior: move towards food if detected
        if food_info['found']:
            # Food detected, move towards it
            x_center = food_info['center_x']
            size = food_info['size']
            
            # Calculate movement based on food position
            if x_center < 0.4:  # Food is on the left
                rob.move(15, 5)  # Turn left while moving forward
                rob.set_emotion(Emotion.HAPPY)
            elif x_center > 0.6:  # Food is on the right
                rob.move(5, 15)  # Turn right while moving forward
                rob.set_emotion(Emotion.HAPPY)
            else:  # Food is centered
                rob.move(20, 20)  # Move forward
                rob.set_emotion(Emotion.SUPER_HAPPY)
                
            print(f"Food detected at x={x_center:.2f}, size={size:.2f}")
        else:
            # No food detected, search by turning
            rob.move(-5, 5)  # Turn in place
            rob.set_emotion(Emotion.NORMAL)
            print("Searching for food...")
        
        # Check if food was collected (simulation only)
        if isinstance(rob, SimulationRobobo):
            current_food = rob.get_nr_food_collected()
            if current_food > food_collected:
                rob.set_emotion(Emotion.LAUGHING)
                rob.play_emotion_sound(SoundEmotion.LAUGHING)
                print(f"Food collected! Total: {current_food}")
                food_collected = current_food
        
        time.sleep(0.1)  # Small delay between iterations
    
    # Stop the robot at the end
    rob.move(0, 0)
    
    if isinstance(rob, SimulationRobobo):
        print(f"\nDemo completed! Food collected: {rob.get_nr_food_collected()}")
    else:
        print("\nDemo completed!")
    
    print("===== DEMO ENDED =====")


def _plot_training_results(results, agent_type, timestamp):
    """Plot and save training results from RL runs"""
    import matplotlib.pyplot as plt
    import numpy as np
    from pathlib import Path
    import os
    
    # Create plots directory if it doesn't exist
    plots_dir = Path("results/figures")
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate plot file name with timestamp
    file_name = f"task2_{agent_type}_training_{timestamp}.png"
    file_path = plots_dir / file_name
    
    # Check if results contains the required data
    if 'episode_rewards' not in results:
        print(f"Warning: No episode_rewards in results, cannot generate training plot")
        return None
    
    # Set up the plot
    plt.figure(figsize=(12, 8))
    
    # Plot episode rewards
    rewards = results['episode_rewards']
    episodes = np.arange(1, len(rewards) + 1)
    plt.subplot(2, 2, 1)
    plt.plot(episodes, rewards, 'b-')
    plt.title(f'Episode Rewards ({agent_type.upper()})')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True)
    
    # Plot moving average of rewards
    window = min(10, len(rewards))
    if window > 0:
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        plt.subplot(2, 2, 2)
        plt.plot(np.arange(window, len(rewards) + 1), moving_avg, 'r-')
        plt.title(f'Moving Average Reward (window={window})')
        plt.xlabel('Episode')
        plt.ylabel('Average Reward')
        plt.grid(True)
    
    # Plot food collection stats if available
    if 'food_collected' in results:
        plt.subplot(2, 2, 3)
        plt.plot(episodes, results['food_collected'], 'g-')
        plt.title('Food Items Collected')
        plt.xlabel('Episode')
        plt.ylabel('Number of Items')
        plt.grid(True)
    
    # Plot exploration rate or other metric if available
    if 'exploration_rates' in results:
        plt.subplot(2, 2, 4)
        plt.plot(episodes, results['exploration_rates'], 'm-')
        plt.title('Exploration Rate')
        plt.xlabel('Episode')
        plt.ylabel('Epsilon')
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(str(file_path))
    plt.close()
    
    print(f"Training plot saved to {file_path}")
    return file_path


def plot_task2_training_progress(results, agent_type='dqn'):
    """Plot and visualize training progress for Task 2 (Food Collection)
    
    This function creates visualizations of the training progress including:
    - Episode rewards
    - Moving average rewards
    - Food collection statistics
    - Exploration rate decay
    
    Args:
        results: Dictionary containing training results
        agent_type: Type of RL agent used ('dqn', 'qlearning', etc.)
        
    Returns:
        Path to the saved plot file
    """
    from datetime import datetime
    
    # Generate timestamp for unique filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Call the internal plotting function
    return _plot_training_results(results, agent_type, timestamp)
