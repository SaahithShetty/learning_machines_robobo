"""
Task 3: Object Pushing to Target Location - Reinforcement Learning Implementation
===============================================================================

TASK 3 IMPLEMENTATION:
- Robot must push a red object to a green target area as fast as possible
- Two difficulty levels: arena_push_easy.ttt and arena_push_hard.ttt
- Focus on object manipulation and spatial reasoning
- Hybrid reward system: vision for navigation + IR for confirmation

SENSOR FUSION APPROACH (Vision + IR):
- VISION: Used for navigation guidance only
  * Detects red objects and green targets
  * Provides distance/angle information for navigation
  * Guides robot towards objects and targets
- IR SENSORS: Used for physical confirmation
  * Front sensors (FrontL, FrontR, FrontC) detect actual contact
  * Confirm object collection and target reaching
  * Trigger phase transitions and task completion

PHASE STRUCTURE:
- Phase 0: Object Detection (vision-based search for red object)
- Phase 1: Object Collection (approach and collect using IR sensors)
- Phase 2: Target Detection (vision-based search and centering for green target)
- Phase 3: Push to Target (vision guides, IR confirms completion)

RECENT UPDATES (Task 3 Transition):
- Updated from Task 2 (food collection) to Task 3 (object pushing)
- Replaced FoodVisionProcessor with ObjectPushVisionProcessor
- New vision system detects red objects and green target areas
- Modified reward system for object pushing task
- Implemented hybrid vision+IR confirmation system
- Same 8-action space maintained for consistency

This module implements Task 3 for the Robobo robot using reinforcement learning
and computer vision. The robot must push a red object into a green target area
using DQN (Deep Q-Network) for intelligent navigation and OpenCV for object/target detection.

Key Components:
- RobotEnvironment: RL environment wrapper optimized for object pushing
- DQNAgent: Deep Q-Network agent for action selection and learning  
- ObjectPushVisionProcessor: Computer vision system for red object and green target detection
- Hybrid sensor fusion: Vision for navigation, IR for confirmation

State Space (11D): [8 IR sensors + 3 vision features]
Action Space (8D): backward, turns (left/right), forward movements
Sensor Order: [BackL, BackR, FrontL, FrontR, FrontC, FrontRR, BackC, FrontLL]

Author: Learning Machines Team
Focus: Task 3 (Object Pushing to Target) - Foraging Task
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
        f"TaskDone:{info.get('task_completed', False)} | "
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
    """Main function to run Task 3: Object Pushing to Target Location
    
    This is the primary entry point for running the object pushing task
    using reinforcement learning and computer vision.
    
    Args:
        rob: Robot interface instance (SimulationRobobo or HardwareRobobo)
        rl_agent_type: Type of RL agent ('dqn' for Task 3)
        rl_mode: RL mode ('train', 'evaluate', or 'train_and_evaluate')
        rl_episodes: Number of RL episodes to run
        
    Returns:
        Dictionary with training results and performance metrics
    """
    if isinstance(rob, SimulationRobobo):
        rob.play_simulation()
    
    rob.set_phone_tilt_blocking(100, 100)
    rob.set_phone_pan_blocking(180, 50) # Center camera for optimal object and target detection
    print("="*50)
    print(f"TASK 3: OBJECT PUSHING TO TARGET USING OPENCV + RL ({rl_agent_type.upper()})")
    print("="*50)
    
    # Run Task 3 - Object Pushing to Target
    results = object_pushing_task3(
        rob, agent_type=rl_agent_type, mode=rl_mode, num_episodes=rl_episodes,
        collision_threshold=collision_threshold
    )
    
    print(f"\nTask 3 completed!")
    if rl_mode in ['train', 'train_and_evaluate']:
        print(f"Models and metrics saved to {FIGURES_DIR}")
    
    if isinstance(rob, SimulationRobobo):
        rob.stop_simulation()
        
    return results


class RobotEnvironment:
    """Environment wrapper for Task 3: Object Pushing to Target Location using Reinforcement Learning
    
    This environment is specifically designed for Task 3, which involves:
    - Pushing a red object to a green target area as fast as possible
    - Using computer vision and IR sensors for object detection and collection
    - Implementing intelligent navigation and manipulation strategies
    - Providing a 14-dimensional state space: [8 IR sensors + 6 vision features]
    - Supporting 8 discrete actions optimized for object pushing
    
    Phase Structure (4 phases):
    - Phase 0: Object Detection (vision-based search for red object)
    - Phase 1: Object Collection (approach and collect using IR sensors)
    - Phase 2: Target Detection (search for green target while carrying object)
    - Phase 3: Push to Target (move object to target location)
    """
    
    def __init__(self, robot: IRobobo, vision_processor, max_episode_time: int = 300, 
                 collision_threshold: float = 0.95, save_images: bool = False, 
                 image_save_interval: int = 100):
        self.robot = robot
        self.vision_processor = vision_processor
        self.max_episode_time = max_episode_time  # 5 minutes for Task 3
        self.step_count = 0
        
        # Image saving configuration
        self.save_images = save_images
        self.image_save_interval = image_save_interval  # Save every N steps
        self.episode_count = 0
        self.images_saved_this_episode = 0
        
        # Create images directory if saving is enabled
        if self.save_images:
            self.images_dir = FIGURES_DIR / "training_images"
            self.images_dir.mkdir(parents=True, exist_ok=True)
            print(f"Image saving enabled: Every {image_save_interval} steps to {self.images_dir}")
        
        # Detect environment type for proper camera positioning
        self.is_simulation = hasattr(self.robot, '_smartphone_camera')
        env_type = "simulation" if self.is_simulation else "hardware"
        print(f"Initializing robot environment for {env_type}")
        
        # Unified collision threshold (same for simulation and hardware)
        # This single threshold is used for ALL IR sensor collision detection
        # Can be configured via terminal script: --collision-threshold 0.95
        self.collision_threshold = collision_threshold
        
        # Phase-specific action sets
        self.use_phase_specific_actions = True  # Enable phase-specific actions
        
        # Universal action set (original)
        self.universal_actions = [
            (-50, -50),  # 0: Backward
            (-30, 30),   # 1: Turn Left
            (-15, 30),   # 2: Turn Left Slight
            (40, 60),    # 3: Forward Left
            (60, 60),    # 4: Forward
            (60, 40),    # 5: Forward Right
            (30, -15),   # 6: Turn Right Slight
            (30, -30),   # 7: Turn Right
        ]
        
        # Phase-specific action sets
        self.phase_actions = {
            0: [  # Phase 0: Object Detection - Systematic right turn search only
                (5, -5),   # 0: Turn Right Very Slight
                (5, -5),   # 0: Turn Right Very Slight
                (5, -5),   # 0: Turn Right Very Slight
                (5, -5),   # 0: Turn Right Very Slight
                (5, -5),   # 0: Turn Right Very Slight
                (5, -5),   # 0: Turn Right Very Slight
                (5, -5),   # 0: Turn Right Very Slight
                (5, -5),   # 0: Turn Right Very Slight
            ],
            1: [  # Phase 1: Object Collection - Forward only when object is detected
                (60, 60),    # 0: Forward
                (60, 60),    # 1: Forward
                (60, 60),    # 2: Forward
                (60, 60),    # 3: Forward
                (60, 60),    # 4: Forward
                (60, 60),    # 5: Forward
                (60, 60),    # 6: Forward
                (60, 60),    # 7: Forward
            ],
            2: [  # Phase 2: Target Detection - Systematic right turn search only
                (10, -10),   # 0: Turn Right Slight
                (10, -10),   # 1: Turn Right Slight
                (10, -10),   # 2: Turn Right Slight
                (10, -10),   # 3: Turn Right Slight
                (10, -10),   # 4: Turn Right Slight
                (10, -10),   # 5: Turn Right Slight
                (10, -10),   # 6: Turn Right Slight
                (10, -10),   # 7: Turn Right Slight
            ],
            3: [  # Phase 3: Push to Target - Forward only after alignment
                (60, 60),    # 0: Forward
                (60, 60),    # 1: Forward
                (60, 60),    # 2: Forward
                (60, 60),    # 3: Forward
                (60, 60),    # 4: Forward
                (60, 60),    # 5: Forward
                (60, 60),    # 6: Forward
                (60, 60),    # 7: Forward
            ]
        }
        
        # Phase action descriptions
        self.phase_action_descriptions = {
            0: ["Turn Right Very Slight", "Turn Right Very Slight", "Turn Right Very Slight", "Turn Right Very Slight", 
                "Turn Right Very Slight", "Turn Right Very Slight", "Turn Right Very Slight", "Turn Right Very Slight"],
            1: ["Forward", "Forward", "Forward", "Forward", "Forward", "Forward", "Forward", "Forward"],
            2: ["Turn Right Slight", "Turn Right Slight", "Turn Right Slight", "Turn Right Slight", 
                "Turn Right Slight", "Turn Right Slight", "Turn Right Slight", "Turn Right Slight"],
            3: ["Forward", "Forward", "Forward", "Forward", 
                "Forward", "Forward", "Forward", "Forward"]
        }
        
        # Set initial action space based on phase
        self.current_phase_num = 0  # Start with phase 0
        self._update_action_space()
        
        # State space: IR sensors (8) + vision data (6) = 14 dimensions
        # Vision data: [object_detected, object_distance, object_angle, target_detected, target_distance, target_angle]
        self.state_size = 14
        
        # Task 3 specific tracking
        self.task_completed = False
        self.episode_start_time = None
        self.last_object_position = None
        self.last_target_position = None
        
        # Sequential task phases for Task 3 (4 phases)
        self.current_phase = "OBJECT_DETECTION"  # Phases: OBJECT_DETECTION -> OBJECT_COLLECTION -> TARGET_DETECTION -> PUSH_TO_TARGET -> COMPLETED
        self.object_collected = False  # Flag for when robot has reached the object
        
        # Object-target relationship tracking
        self.object_target_distance_history = deque(maxlen=10)  # Track object-target proximity
        
        # Collision state tracking for object interaction
        self.last_collision_step = -1
        self.collision_cooldown = 3  # Prevent multiple interactions within N steps
        
        # Action history for temporal learning and grace periods
        self.action_history = deque(maxlen=10)  # Track last 10 actions
        self.reward_history = deque(maxlen=10)  # Track last 10 rewards
        self.grace_period_steps = 3  # Steps to wait before penalizing certain actions
        
        # Phase 1 -> Phase 2 transition tracking using raw IR sensors
        self.front_ir_history = []  # Track ALL front IR distance readings per episode
        self.phase_transition_threshold_mm = 5.9  # 6mm threshold for object contact
        self.required_close_readings = 20  # Need 8+ readings below threshold for transition
        
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
        self.cached_vision_data = {
            'object_detected': 0.0, 'object_distance': 1.0, 'object_angle': 0.0,
            'target_detected': 0.0, 'target_distance': 1.0, 'target_angle': 0.0
        }

        self.reset()

    def _update_action_space(self):
        """Update action space based on current phase"""
        if self.use_phase_specific_actions and self.current_phase_num in self.phase_actions:
            self.actions = self.phase_actions[self.current_phase_num]
            self.action_space_size = len(self.actions)
            self.action_descriptions = self.phase_action_descriptions[self.current_phase_num]
            print(f"Phase {self.current_phase_num}: Updated to {self.action_space_size} actions: {self.action_descriptions}")
        else:
            # Fallback to universal actions
            self.actions = self.universal_actions
            self.action_space_size = len(self.actions)
            self.action_descriptions = [
                "Backward", "Turn Left", "Turn Left Slight", "Forward Left", 
                "Forward", "Forward Right", "Turn Right Slight", "Turn Right"
            ]

    def _transition_to_phase(self, new_phase: int, reason: str = ""):
        """Transition to a new phase and update action space"""
        if new_phase != self.current_phase_num:
            old_phase = self.current_phase_num
            self.current_phase_num = new_phase
            self.phase_history.append((old_phase, new_phase, self.step_count, reason))
            self._update_action_space()
            print(f"Phase transition: {old_phase} -> {new_phase} (Step {self.step_count}): {reason}")

    def get_action_space_size(self):
        """Get current action space size (for compatibility with RL agents)"""
        return self.action_space_size
        
    def get_current_phase(self):
        """Get current phase number"""
        return self.current_phase_num
    
    def reset(self):
        """Reset environment for new episode"""
        self.step_count = 0
        
        if self.is_simulation:
            # Reset robot position in simulation
            self.robot.stop_simulation()
            time.sleep(0.5)
            self.robot.play_simulation()
            time.sleep(1.0)
        
        # Initialize camera for optimal object and target detection
        self._initialize_camera()
        
        # Reset Task 3 specific counters
        self.task_completed = False
        self.current_phase_num = 0  # Reset to phase 0
        self.object_collected = False
        self.target_centered = False
        self.phase_history = []
        
        # Update episode count and reset image counter
        self.episode_count += 1
        self.images_saved_this_episode = 0
        
        # Update action space for initial phase
        self._update_action_space()
        
        # Use simulation time for simulator, real time for hardware
        if hasattr(self.robot, 'get_sim_time'):
            # Simulation robot - use simulation time for proper headless speedup
            self.episode_start_time = self.robot.get_sim_time()
        else:
            # Hardware robot - use real time
            self.episode_start_time = time.time()
        
        # Reset collision state tracking
        self.last_collision_step = -1
        
        # Reset object-target tracking
        self.last_object_position = None
        self.last_target_position = None
        self.last_target_distance = 1.0  # Initialize for Phase 2 progress tracking
        self.object_target_distance_history.clear()
        
        # Reset IR history for phase transition tracking
        self.front_ir_history.clear()  # Clear all previous episode readings
        
        # Reset Phase 3 task completion tracking
        self.green_disappearance_count = 0
        self.last_green_seen_step = 0
        
        # Debug: Check initial state and camera setup
        self._debug_camera_position()
        initial_state = self._get_state()
        ir_raw = self.robot.read_irs()
        print(f"Reset - IR readings: {ir_raw[:4]}, Task completed: {self.task_completed}")
        
        return initial_state
    
    def _initialize_camera(self):
        """Initialize camera - tilt downward to see objects on the ground"""
        try:
            # For Task 3 (object pushing), we need the camera to look down at the ground
            # to detect red objects and green targets
            print(f"Camera initial position - Pan: {self.robot.read_phone_pan()}, Tilt: {self.robot.read_phone_tilt()}")
            
            # Tilt camera downward to see objects on the ground
            # Negative tilt values point the camera downward
            target_tilt = 250  # Tilt down 30 degrees to see ground objects
            self.robot.set_phone_tilt_blocking(target_tilt, 500)  # Move over 0.5 seconds
            
            # Small delay to ensure tilt movement completes
            time.sleep(1.0)
            
            # Verify camera position
            final_pan = self.robot.read_phone_pan()
            final_tilt = self.robot.read_phone_tilt()
            print(f"Camera positioned for object detection - Pan: {final_pan}, Tilt: {final_tilt}")
            
        except Exception as e:
            print(f"Warning: Could not set camera position: {e}")
            print("Continuing with default camera position...")

    def _debug_camera_position(self):
        """Debug camera positioning and object/target detection"""
        try:
            print(f"\n🔍 Camera Debug Info:")
            print(f"  Current Pan: {self.robot.read_phone_pan()}")
            print(f"  Current Tilt: {self.robot.read_phone_tilt()}")
            
            # Test if camera is working
            camera_frame = self.robot.read_image_front()
            if camera_frame is not None:
                print(f"  Camera frame size: {camera_frame.shape}")
                
                # Test object and target detection
                red_objects, green_targets, _ = self.vision_processor.detect_objects_and_target(camera_frame)
                print(f"  Red objects detected: {len(red_objects)}")
                print(f"  Green targets detected: {len(green_targets)}")
                
                for i, obj in enumerate(red_objects[:3]):  # Show first 3
                    print(f"    Red {i+1}: Distance={obj['distance']:.3f}, Angle={obj['angle']:.1f}°")
                    
                for i, target in enumerate(green_targets[:3]):  # Show first 3
                    print(f"    Green {i+1}: Distance={target['distance']:.3f}, Angle={target['angle']:.1f}°")
            else:
                print("  ❌ No camera frame received!")
                
        except Exception as e:
            print(f"  Camera debug error: {e}")

    def _save_camera_image_if_needed(self):
        """Save camera image if image saving is enabled and it's time to save"""
        if not self.save_images:
            return
            
        # Save every N steps
        if self.step_count % self.image_save_interval == 0:
            try:
                # Get current camera frame
                camera_frame = self.robot.read_image_front()
                if camera_frame is not None:
                    # Create filename with episode, step, and phase info
                    filename = f"ep{self.episode_count:03d}_step{self.step_count:04d}_phase{self.current_phase_num}.jpg"
                    filepath = self.images_dir / filename
                    
                    # Get vision data for annotation
                    red_objects, green_targets, processed_frame = self.vision_processor.detect_objects_and_target(camera_frame)
                    
                    # Save both original and processed frame
                    cv2.imwrite(str(filepath), camera_frame)
                    
                    # Save processed frame with detections if available
                    if processed_frame is not None:
                        processed_filename = f"ep{self.episode_count:03d}_step{self.step_count:04d}_phase{self.current_phase_num}_processed.jpg"
                        processed_filepath = self.images_dir / processed_filename
                        cv2.imwrite(str(processed_filepath), processed_frame)
                    
                    self.images_saved_this_episode += 1
                    
                    # Print info occasionally
                    if self.step_count <= 10 or self.step_count % (self.image_save_interval * 5) == 0:
                        print(f"  📸 Saved image: {filename} (Red: {len(red_objects)}, Green: {len(green_targets)})")
                        
            except Exception as e:
                print(f"Warning: Could not save camera image: {e}")
                
    def _get_state(self):
        """Get current state representation for Task 3: [IR sensors + vision]"""
        # IR sensors (8 dimensions) - Correct order: [BackL, BackR, FrontL, FrontR, FrontC, FrontRR, BackC, FrontLL]
        ir_values = self.robot.read_irs()
        ir_normalized = []
        for val in ir_values:
            if val is None:
                ir_normalized.append(1.0)  # No detection = maximum distance
            else:
                # FIXED: Proper IR distance calculation
                # IR values are intensity values, need to convert to distance
                distance_mm = self.vision_processor._ir_intensity_to_distance(val)
                # Normalize to [0,1] where 0=very close, 1=far/no detection
                ir_normalized.append(min(distance_mm / 1000.0, 1.0))
        
        # Vision data (6 dimensions: object_detected, object_distance, object_angle, target_detected, target_distance, target_angle)
        # Use object-target relationship detection for comprehensive spatial awareness
        vision_data = self._get_object_target_state()
        object_detected = vision_data[0]    # Binary: red object visible
        object_distance = vision_data[1]    # Normalized distance [0,1]
        object_angle = vision_data[2]       # Normalized angle [-1,1]
        target_detected = vision_data[3]    # Binary: green target visible
        target_distance = vision_data[4]    # Normalized distance [0,1]
        target_angle = vision_data[5]       # Normalized angle [-1,1]
        
        # Combine: 8 IR + 6 vision = 14 dimensions
        state = np.array(ir_normalized + [object_detected, object_distance, object_angle, 
                                         target_detected, target_distance, target_angle], 
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
        self.robot.move_blocking(left_speed, right_speed, miniduration)

        # Save camera image periodically for analysis
        self._save_camera_image_if_needed()

        # Get new state
        next_state = self._get_state()
        
        # Calculate reward and determine emotion
        reward, info = self._calculate_reward(action_idx, next_state)
        
        # Debug: Print reward info occasionally
        if self.step_count <= 10 or self.step_count % 50 == 0:
            print(f"  Reward: {reward:.2f}, Task completed: {self.task_completed}, Info: {info}")
        
        # Set emotion based on reward and situation
        if info.get('task_completed', False):
            emotion = Emotion.HAPPY
        elif info.get('obstacle_collision', False):
            emotion = Emotion.SAD
        elif info.get('object_detected', False) and info.get('target_detected', False):
            emotion = Emotion.SURPRISED  # Excited about seeing both object and target
        elif reward > 10.0:
            emotion = Emotion.HAPPY
        elif reward < -5.0:
            emotion = Emotion.ANGRY
        else:
            emotion = Emotion.NORMAL
        
        # Store emotion and additional info
        info['emotion'] = emotion
        info['action_taken'] = self.action_descriptions[action_idx]
        info['task_completed'] = self.task_completed
        
        # Check if episode is done - Task 3: Time limit or task completion
        # Use appropriate time measurement based on robot type
        if hasattr(self.robot, 'get_sim_time'):
            # Simulation robot - use simulation time for proper headless speedup
            time_elapsed = self.robot.get_sim_time() - self.episode_start_time if self.episode_start_time else 0
        else:
            # Hardware robot - use real time
            time_elapsed = time.time() - self.episode_start_time if self.episode_start_time else 0
            
        done = (
            time_elapsed >= self.max_episode_time or
            self.task_completed
        )
        
        # Add episode completion reward/penalty
        if done:
            if self.task_completed:
                # Success: Object pushed to target
                time_remaining = max(0, self.max_episode_time - time_elapsed)
                success_bonus = 500 + (time_remaining / self.max_episode_time) * 200  # Big bonus for success + speed
                reward += success_bonus
                print(f"🏆 EPISODE SUCCESS! Object pushed to target! Bonus: {success_bonus:.1f}")
            elif time_elapsed >= self.max_episode_time:
                # Failure: Time ran out
                failure_penalty = -200  # Penalty for not completing in time
                reward += failure_penalty
                print(f"⏰ EPISODE TIMEOUT! Task not completed. Penalty: {failure_penalty}")
            
            info['episode_complete'] = True
            info['success'] = self.task_completed
        
        # Check for task completion
        task_completed_this_step = self._check_task_completion()
        if task_completed_this_step:
            self.task_completed = True
            info['task_completed'] = True
            print(f"🎉 TASK COMPLETED! Object successfully pushed to target!")
        
        # Note: The reward calculation in _calculate_reward already handles 
        # object pushing progress and sets relevant info
        
        return next_state, reward, done, info
    
    def _calculate_reward(self, action_idx: int, state: np.ndarray) -> Tuple[float, dict]:
        """Phase-wise reward function for Task 3: Object Pushing to Target"""
        reward = 0.0
        info = {}
        
        # Basic info setup
        ir_values = list(state[:8])
        vision_values = list(state[8:])
        
        # Extract vision data
        object_detected = vision_values[0]    # Binary: red object visible
        object_distance = vision_values[1]    # Normalized distance [0,1]
        object_angle = vision_values[2]       # Normalized angle [-1,1]
        target_detected = vision_values[3]    # Binary: green target visible
        target_distance = vision_values[4]    # Normalized distance [0,1]
        target_angle = vision_values[5]       # Normalized angle [-1,1]
        
        # Phase-wise reward calculation
        if self.current_phase_num == 0:
            # Phase 0: Object Detection (vision-based search)
            reward = self._calculate_phase0_reward(action_idx, ir_values, vision_values, info)
        elif self.current_phase_num == 1:
            # Phase 1: Object Collection (approach and collect)
            reward = self._calculate_phase1_reward(action_idx, ir_values, vision_values, info)
        elif self.current_phase_num == 2:
            # Phase 2: Target Detection (search for green target)
            reward = self._calculate_phase2_reward(action_idx, ir_values, vision_values, info)
        elif self.current_phase_num == 3:
            # Phase 3: Push to Target (move object to target)
            reward = self._calculate_phase3_reward(action_idx, ir_values, vision_values, info)
        else:
            # Fallback: Basic forward movement reward
            if action_idx == 4:  # Forward action
                reward += 1.0
                info['forward_movement'] = True
        
        # Store basic info
        info['phase'] = self.current_phase_num
        info['action_description'] = self.action_descriptions[action_idx]
        info['object_detected'] = object_detected > 0.5
        info['target_detected'] = target_detected > 0.5
        
        return reward, info

    def _calculate_phase0_reward(self, action_idx, ir_values, vision_values, info):
        """Phase 0: Object Detection - Vision-based search for red object, fallback to front IR if not visible"""
        reward = 0.0
        # Extract vision data
        object_detected = vision_values[0]    # Binary: red object visible
        object_distance = vision_values[1]    # Normalized distance [0,1]
        object_angle = vision_values[2]       # Normalized angle [-1,1]
        # Extract front center IR sensor only
        front_center_ir = ir_values[4]  # FrontC only
        
        # Debug output for Phase 0 vision detection issues
        if self.step_count <= 20 or self.step_count % 10 == 0:
            print(f"  Phase 0 Debug - Vision: detected={object_detected:.3f}, angle={object_angle:.3f}, distance={object_distance:.3f}, FrontC_IR={front_center_ir:.6f}")
            
            # Save debug image every few steps in Phase 0
            try:
                camera_frame = self.robot.read_image_front()
                if camera_frame is not None and self.save_images:
                    # Force save image for debugging vision issues
                    filename = f"DEBUG_phase0_step{self.step_count}_vision{object_detected:.1f}_angle{object_angle:.2f}.jpg"
                    filepath = self.images_dir / filename
                    
                    # Get processed frame with detections
                    red_objects, green_targets, processed_frame = self.vision_processor.detect_objects_and_target(camera_frame)
                    
                    # Save both original and processed
                    cv2.imwrite(str(filepath), camera_frame)
                    if processed_frame is not None:
                        processed_filepath = self.images_dir / f"DEBUG_phase0_step{self.step_count}_PROCESSED.jpg"
                        cv2.imwrite(str(processed_filepath), processed_frame)
                    
                    print(f"    DEBUG: Saved images - Raw objects: {len(red_objects)}, Raw targets: {len(green_targets)}")
            except Exception as e:
                print(f"    DEBUG: Image save failed: {e}")
        
        # 1. Vision-Based Detection
        if object_detected > 0.5:
            reward += 10.0  # Base reward for vision detection
            abs_angle = abs(object_angle)
            if abs_angle <= 0.20:
                reward += 10.0  # Centered
            elif abs_angle <= 0.66:
                reward += 3.0   # Side
            distance_reward = (1.0 - object_distance) * 5.0
            reward += distance_reward
            info['object_in_view'] = True
            
            # DEBUG: Save image when red object is detected
            try:
                camera_frame = self.robot.read_image_front()
                if camera_frame is not None and hasattr(self, 'images_dir'):
                    # Apply image correction for reversed phone camera
                    # Phone is upside down, so flip both horizontally and vertically
                    corrected_frame = cv2.flip(camera_frame, -1)  # -1 = flip both axes (180° rotation)
                    
                    # Save original camera frame (uncorrected)
                    debug_filename = f"HARDWARE_PHASE0_DETECTION_step{self.step_count}_obj{object_detected:.2f}_angle{object_angle:.3f}_dist{object_distance:.3f}_ORIGINAL.jpg"
                    debug_filepath = self.images_dir / debug_filename
                    cv2.imwrite(str(debug_filepath), camera_frame)
                    
                    # Save corrected camera frame
                    corrected_filename = f"HARDWARE_PHASE0_DETECTION_step{self.step_count}_obj{object_detected:.2f}_angle{object_angle:.3f}_dist{object_distance:.3f}_CORRECTED.jpg"
                    corrected_filepath = self.images_dir / corrected_filename
                    cv2.imwrite(str(corrected_filepath), corrected_frame)
                    
                    # Get and save processed detection overlay
                    try:
                        object_target_info = self.vision_processor.get_object_target_info_dict(corrected_frame)
                        processed_frame = corrected_frame.copy()
                        
                        # Draw detection results on frame
                        if object_target_info.get('red_objects_found', False):
                            red_x = object_target_info.get('red_center_x', 0.5)
                            red_y = object_target_info.get('red_center_y', 0.5)
                            h, w = processed_frame.shape[:2]
                            center_x = int(red_x * w)
                            center_y = int(red_y * h)
                            
                            # Draw red circle at detected object center
                            cv2.circle(processed_frame, (center_x, center_y), 20, (0, 0, 255), 3)
                            cv2.putText(processed_frame, f"RED: ({red_x:.2f},{red_y:.2f})", 
                                      (center_x + 25, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        
                        # Save processed frame with overlay
                        processed_filename = f"HARDWARE_PHASE0_DETECTION_step{self.step_count}_obj{object_detected:.2f}_angle{object_angle:.3f}_dist{object_distance:.3f}_PROCESSED.jpg"
                        processed_filepath = self.images_dir / processed_filename
                        cv2.imwrite(str(processed_filepath), processed_frame)
                        
                        print(f"HARDWARE DEBUG - Saved detection images: {debug_filename}, {corrected_filename}, {processed_filename}")
                        
                    except Exception as e:
                        print(f"HARDWARE DEBUG - Failed to create processed overlay: {e}")
                        
            except Exception as e:
                print(f"HARDWARE DEBUG - Failed to save detection image: {e}")
            
            # Transition if well centered and close enough
            if abs_angle <= 0.17 and object_distance <= 0.5:
                info['phase_transition_ready'] = True
                self._transition_to_phase(1, f"Object detected and positioned (angle={object_angle:.3f}, distance={object_distance:.3f}) - ready for collection")
            # Early transition if very well centered (prevent overshooting)
            elif abs_angle <= 0.15:
                info['phase_transition_ready'] = True
                self._transition_to_phase(1, f"Object well centered (angle={object_angle:.3f}) - transition to prevent overshooting")
        else:
            info['object_in_view'] = False
            
        # Encourage systematic exploration if nothing detected
        if not info['object_in_view']:
            # All actions in Phase 0 are right turns, so encourage systematic scanning
            reward += 2.0  # Base turning reward for systematic search
        else:
            # If object detected, discourage continued turning to prevent overshooting
            # Give smaller reward to encourage stopping when object is well positioned
            if info['object_in_view'] and abs(object_angle) <= 0.2:
                reward += 1.0  # Small reward to encourage stopping when well positioned
            else:
                reward += 2.0  # Normal turning reward
        info['object_angle'] = object_angle
        info['object_distance'] = object_distance
        info['front_center_ir'] = front_center_ir
        return reward

    def _calculate_phase1_reward(self, action_idx, ir_values, vision_values, info):
        """Phase 1: Object Collection - Move straight, use vision + IR for robust transition"""
        reward = 0.0
        
        # Extract vision data for transition detection
        object_detected = vision_values[0]    # Binary: red object visible
        object_distance = vision_values[1]    # Normalized distance [0,1]
        object_angle = vision_values[2]       # Normalized angle [-1,1]
        
        # Get raw IR values directly from robot for backup detection
        raw_ir_values = self.robot.read_irs()
        front_center_ir_intensity = raw_ir_values[4]  # Raw intensity from FrontC sensor
        
        # Convert raw IR intensity to actual distance
        front_center_distance_mm = self.vision_processor._ir_intensity_to_distance(front_center_ir_intensity)
        
        # Add current distance reading to history (collect ALL readings per episode)
        self.front_ir_history.append(front_center_distance_mm)
        
        # Count readings below 6mm threshold
        close_readings_count = sum(1 for distance in self.front_ir_history if distance <= self.phase_transition_threshold_mm)
        
        # Debug output for Phase 1 collection issues
        if self.step_count <= 20 or self.step_count % 10 == 0:
            print(f"  Phase 1 Debug - Vision: detected={object_detected:.3f}, distance={object_distance:.3f}, angle={object_angle:.3f}")
            print(f"  Phase 1 Debug - IR: Distance={front_center_distance_mm:.1f}mm, Raw_IR={front_center_ir_intensity:.6f}")
            print(f"  Phase 1 Debug - Close_readings={close_readings_count}/{len(self.front_ir_history)}, Need={self.required_close_readings} for IR transition")
        
        # Reward for moving forward (all actions in Phase 1 are forward)
        if action_idx in [0,1,2,3,4,5,6,7]:
            reward += 5.0
        
        # Progressive reward for getting closer to object (vision-based)
        if object_detected > 0.5:
            # Object is visible - use vision for distance estimation
            distance_reward = (1.0 - object_distance) * 15.0  # Higher reward for vision-based distance
            reward += distance_reward
            info['object_visible_in_phase1'] = True
            
            # Centering reward - encourage keeping object centered while approaching
            abs_angle = abs(object_angle)
            if abs_angle <= 0.1:  # Very well centered
                reward += 10.0
            elif abs_angle <= 0.2:  # Reasonably centered
                reward += 5.0
                
        # Progressive reward for getting closer to object (IR-based backup)
        if front_center_distance_mm < 500:  # Within 50cm
            # Higher reward for closer distances
            distance_reward = (500 - front_center_distance_mm) / 500.0 * 10.0
            reward += distance_reward
        
        # PRIMARY PHASE TRANSITION: Vision-based position detection (corrected camera)
        # When red box appears in bottom section of image = robot has collected it
        if object_detected > 0.5:
            # Get the actual object position from vision processor for position-based transition
            try:
                camera_frame = self.robot.read_image_front()
                if camera_frame is not None:
                    # Apply image correction for reversed phone camera
                    corrected_frame = cv2.flip(camera_frame, -1)  # -1 = flip both axes (180° rotation)
                    
                    # Get object position info
                    object_target_info = self.vision_processor.get_object_target_info_dict(corrected_frame)
                    
                    if object_target_info.get('red_objects_found', False):
                        red_center_y = object_target_info.get('red_center_y', 0.5)  # Normalized Y position [0,1]
                        
                        # Bottom section check: if object is in bottom 15% of image (Y > 0.85) - tuned threshold
                        if red_center_y > 0.85:  # Bottom section of image - robot is touching object
                            reward += 100.0  # Big reward for successful collection via position
                            info['collection_complete'] = True
                            info['phase_transition_ready'] = True
                            info['transition_method'] = 'vision_position'
                            info['object_y_position'] = red_center_y
                            self._transition_to_phase(2, f"Object collected via VISION POSITION - Y={red_center_y:.3f} (bottom section - robot touching) (READY FOR TARGET SEARCH)")
                        
                        # Progress reward for getting close to bottom
                        elif red_center_y > 0.7:  # Bottom 30% of image - getting very close
                            reward += 50.0
                            info['object_very_close'] = True
                            info['object_y_position'] = red_center_y
                        
                        # Progress reward for getting to bottom half
                        elif red_center_y > 0.6:  # Bottom 40% of image - approaching
                            reward += 20.0
                            info['object_approaching_bottom'] = True
                            info['object_y_position'] = red_center_y
                            
                        info['object_y_position'] = red_center_y
                        
            except Exception as e:
                print(f"Vision position check failed: {e}")
                # Fall back to distance-based check
                if object_distance <= 0.15:  # Very close fallback
                    reward += 80.0
                    info['collection_complete'] = True
                    info['phase_transition_ready'] = True
                    info['transition_method'] = 'vision_distance_fallback'
                    self._transition_to_phase(2, f"Object collected via VISION DISTANCE FALLBACK - distance={object_distance:.3f} (READY FOR TARGET SEARCH)")
        
        # BACKUP PHASE TRANSITION: IR-based detection (if vision completely fails)
        elif close_readings_count >= self.required_close_readings:
            reward += 60.0  # Lower reward for IR-based transition
            info['collection_complete'] = True
            info['phase_transition_ready'] = True
            info['transition_method'] = 'ir_backup'
            self._transition_to_phase(2, f"Object collected via IR BACKUP - {close_readings_count}/{len(self.front_ir_history)} readings ≤{self.phase_transition_threshold_mm}mm (READY FOR TARGET SEARCH)")
            
        # Progress rewards
        elif object_detected > 0.5 and object_distance <= 0.3:  # Getting close via vision
            reward += 40.0
            info['close_via_vision'] = True
        elif close_readings_count >= self.required_close_readings // 2:  # Half way there via IR
            reward += 20.0
            info['progress_to_collection'] = True
        elif front_center_distance_mm <= 100.0:  # Close but not consistent via IR
            reward += 10.0
            info['close_to_object'] = True
            
        # Store values for debugging
        info['object_detected_vision'] = object_detected > 0.5
        info['object_distance_vision'] = object_distance
        info['object_angle_vision'] = object_angle
        info['front_c_distance_mm'] = front_center_distance_mm
        info['front_c_ir_intensity'] = front_center_ir_intensity
        info['close_readings_count'] = close_readings_count
        info['total_readings'] = len(self.front_ir_history)
        return reward

    def _calculate_phase2_reward(self, action_idx, ir_values, vision_values, info):
        """Phase 2: Target Detection - Search for green target while having object"""
        reward = 0.0
        
        # Extract vision data
        object_detected = vision_values[0]    # Binary: red object visible
        target_detected = vision_values[3]    # Binary: green target visible
        target_distance = vision_values[4]    # Normalized distance [0,1]
        target_angle = vision_values[5]       # Normalized angle [-1,1]
        
        # SAFETY CHECK: If red object is lost, fall back to Phase 0
        if object_detected <= 0.5:
            # Initialize loss tracking if needed
            if not hasattr(self, 'object_loss_count'):
                self.object_loss_count = 0
            
            self.object_loss_count += 1
            
            # Fall back to Phase 0 if object lost for several consecutive steps
            if self.object_loss_count >= 3:  # Lost for 3 consecutive steps
                reward += -20.0  # Penalty for losing object
                info['object_lost_fallback'] = True
                self._transition_to_phase(0, f"RED OBJECT LOST in Phase 2 - falling back to Phase 0 for re-detection (lost for {self.object_loss_count} steps)")
                return reward
        else:
            # Reset loss counter if object is visible
            self.object_loss_count = 0
        
        # 1. Target Detection Rewards
        if target_detected > 0.5:  # Green target is visible
            # Base reward for finding target
            base_target_reward = 10.0
            reward += base_target_reward
            info['target_in_view'] = True
            
            # Position-based rewards (same 3-section approach as Phase 0)
            abs_angle = abs(target_angle)
            
            if abs_angle <= 0.33:  # Target in middle section
                middle_reward = 15.0
                reward += middle_reward
                info['target_in_middle'] = True
            elif abs_angle <= 0.66:  # Target in side sections
                side_reward = 5.0
                reward += side_reward
                info['target_in_side'] = True
            
            # Distance-based reward (closer = better)
            distance_reward = (1.0 - target_distance) * 8.0
            reward += distance_reward
            info['target_distance_reward'] = distance_reward
            
            # Check for phase completion: target must be very centered (strict alignment)
            if abs_angle <= 0.1:  # Target very centered - ready for Phase 2 (stricter than 0.33)
                positioning_bonus = 30.0
                reward += positioning_bonus
                info['target_positioned'] = True
                info['phase_transition_ready'] = True
                
                # Trigger phase transition to Phase 3
                self._transition_to_phase(3, f"Target very centered (angle={target_angle:.3f}) - ready for pushing")
        else:
            info['target_in_view'] = False
        
        # 2. Turning and Searching Rewards
        # All actions in Phase 1 are turning actions for 360-degree search
        if target_detected <= 0.5:  # No target detected - encourage systematic search
            # Base turning reward for systematic search
            base_turning_reward = 2.0
            reward += base_turning_reward
            info['systematic_search'] = True
            
            # Bonus for varied turning (avoid getting stuck)
            if hasattr(self, 'action_history') and len(self.action_history) > 0:
                if action_idx != self.action_history[-1]:  # Different from last action
                    variety_bonus = 1.0
                    reward += variety_bonus
                    info['search_variety'] = True
        
        # 3. Maintain Object Proximity (ensure we don't lose the collected object)
        object_detected = vision_values[0]
        front_c_ir = ir_values[4]  # FrontC sensor
        
        if front_c_ir < 0.3:  # Object still close (maintain collection)
            object_maintenance_reward = 3.0
            reward += object_maintenance_reward
            info['object_maintained'] = True
        elif front_c_ir > 0.6:  # Object too far - penalty
            object_loss_penalty = -5.0
            reward += object_loss_penalty
            info['object_lost'] = True
        
        # 4. Prevent excessive spinning in one direction
        if hasattr(self, 'action_history') and len(self.action_history) >= 3:
            recent_actions = list(self.action_history)[-3:]
            if len(set(recent_actions)) == 1:  # Same action repeated 3 times
                repetition_penalty = -2.0
                reward += repetition_penalty
                info['repetitive_search'] = True
        
        # Store additional info for debugging
        info['target_angle'] = target_angle
        info['target_distance'] = target_distance
        info['front_c_ir'] = front_c_ir
        
        return reward

    def _calculate_phase3_reward(self, action_idx, ir_values, vision_values, info):
        """Phase 3: Push to Target - Navigate and push object to target"""
        reward = 0.0
        
        # Extract vision data
        object_detected = vision_values[0]    # Binary: red object visible
        target_detected = vision_values[3]    # Binary: green target visible
        target_distance = vision_values[4]    # Normalized distance [0,1]
        target_angle = vision_values[5]       # Normalized angle [-1,1]
        
        # Extract front center IR for object maintenance
        front_center_ir = ir_values[4]  # FrontC
        
        # SAFETY CHECK: If red object is lost, fall back to Phase 0
        if object_detected <= 0.5:
            # Initialize loss tracking if needed
            if not hasattr(self, 'object_loss_count_phase3'):
                self.object_loss_count_phase3 = 0
            
            self.object_loss_count_phase3 += 1
            
            # Fall back to Phase 0 if object lost for several consecutive steps
            if self.object_loss_count_phase3 >= 5:  # Lost for 5 consecutive steps (more lenient in Phase 3)
                reward += -30.0  # Higher penalty for losing object in final phase
                info['object_lost_fallback'] = True
                self._transition_to_phase(0, f"RED OBJECT LOST in Phase 3 - falling back to Phase 0 for re-detection (lost for {self.object_loss_count_phase3} steps)")
                return reward
        else:
            # Reset loss counter if object is visible
            self.object_loss_count_phase3 = 0
        
        # 1. Forward Movement Rewards
        # Encourage only straight forward movement since alignment was done in Phase 1
        if action_idx == 1:  # Forward (straight)
            forward_reward = 15.0
            reward += forward_reward
            info['forward_movement'] = True
        elif action_idx == 3:  # Fast Forward (decisive pushing)
            fast_forward_reward = 20.0  # Highest reward for aggressive pushing
            reward += fast_forward_reward
            info['fast_forward'] = True
        elif action_idx == 6:  # Medium Forward (steady push)
            medium_forward_reward = 12.0
            reward += medium_forward_reward
            info['medium_forward'] = True
        # No rewards for other movements since robot should already be aligned
        
        # 2. Distance-Based Rewards: Green Target Area Coverage
        if target_detected > 0.5:
            # Base reward for keeping target in view
            target_view_reward = 5.0
            reward += target_view_reward
            info['target_in_view'] = True
            
            # Distance-based reward: closer target = higher reward
            # target_distance: 1.0 = far, 0.0 = very close
            # Convert to proximity: 1.0 - target_distance
            target_proximity = 1.0 - target_distance
            
            # Area coverage reward (proximity squared to emphasize getting very close)
            area_coverage_reward = target_proximity * target_proximity * 20.0  # Max 20 points when very close
            reward += area_coverage_reward
            info['area_coverage_reward'] = area_coverage_reward
            info['target_proximity'] = target_proximity
            
            # Bonus for very close proximity (simulating large area coverage)
            if target_distance <= 0.2:  # Very close to target
                close_proximity_bonus = 25.0
                reward += close_proximity_bonus
                info['very_close_to_target'] = True
                
                # Check for task completion when very close
                if target_distance <= 0.1:  # Extremely close - task completion
                    completion_bonus = 100.0
                    reward += completion_bonus
                    info['task_completion_imminent'] = True
            
            # Directional alignment reward (target should be centered for straight push)
            abs_angle = abs(target_angle)
            if abs_angle <= 0.2:  # Target well-centered
                alignment_reward = 10.0
                reward += alignment_reward
                info['target_aligned'] = True
            elif abs_angle <= 0.4:  # Reasonably centered
                partial_alignment_reward = 5.0
                reward += partial_alignment_reward
                info['target_partially_aligned'] = True
        else:
            # Penalty for losing sight of target
            target_lost_penalty = -8.0
            reward += target_lost_penalty
            info['target_lost'] = True
        
        # 3. Object Maintenance (ensure we're still pushing the object)
        if front_center_ir < 0.3:
            # Object still close - good for maintaining contact while pushing
            object_contact_reward = 6.0
            reward += object_contact_reward
            info['object_contact_maintained'] = True
        elif front_center_ir > 0.7:
            # Object too far - penalty for losing contact
            object_contact_penalty = -10.0
            reward += object_contact_penalty
            info['object_contact_lost'] = True
        
        # 4. Progress tracking: reward for sustained forward movement toward target
        if hasattr(self, 'last_target_distance'):
            distance_change = self.last_target_distance - target_distance
            if distance_change > 0.01:  # Getting closer to target
                progress_reward = distance_change * 50.0  # Reward proportional to progress
                reward += progress_reward
                info['progress_toward_target'] = distance_change
        
        # Store current distance for next step comparison
        self.last_target_distance = target_distance
        
        # Store additional info for debugging
        info['target_distance'] = target_distance
        info['target_angle'] = target_angle
        info['front_center_ir'] = front_center_ir
        
        return reward
        
  
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

    def _get_object_target_state(self):
        """Get object and target state from vision processor for Task 3
        
        Returns:
            list: [object_detected, object_distance, object_angle, target_detected, target_distance, target_angle]
                - object_detected: 1.0 if red object visible, 0.0 otherwise
                - object_distance: normalized distance to object [0,1] where 0=close, 1=far
                - object_angle: normalized angle to object [-1,1] where -1=left, 0=center, 1=right
                - target_detected: 1.0 if green target visible, 0.0 otherwise  
                - target_distance: normalized distance to target [0,1] where 0=close, 1=far
                - target_angle: normalized angle to target [-1,1] where -1=left, 0=center, 1=right
        """
        try:
            camera_frame = self.robot.read_image_front()
            if camera_frame is None:
                print("WARNING: Camera frame is None - hardware camera issue?")
                return [0.0, 1.0, 0.0, 0.0, 1.0, 0.0]  # Default: nothing detected, max distance, centered
            
            # Apply image correction for phone camera orientation (needed for both hardware and simulation)
            corrected_frame = cv2.flip(camera_frame, -1)  # 180° rotation for reversed phone camera
            
            # Get object-target relationship from vision processor
            object_target_info = self.vision_processor.get_object_target_info_dict(corrected_frame)
            
            # Hardware debugging - log raw detection results
            if self.current_phase_num == 0 and (self.step_count <= 30 or self.step_count % 20 == 0):
                red_found = object_target_info.get('red_objects_found', False)
                green_found = object_target_info.get('green_targets_found', False)
                print(f"HARDWARE DEBUG - Raw detection: Red={red_found}, Green={green_found}")
                print(f"                  Frame shape: {camera_frame.shape if camera_frame is not None else 'None'}")
                print(f"                  Object info keys: {list(object_target_info.keys())}")
            
            # Extract red object information
            object_detected = 1.0 if object_target_info.get('red_objects_found', False) else 0.0
            object_distance = object_target_info.get('red_distance', 1.0)  # Default to max distance if not found
            object_angle = object_target_info.get('red_angle', 0.0)  # Default to center if not found
            
            # Extract green target information  
            target_detected = 1.0 if object_target_info.get('green_targets_found', False) else 0.0
            target_distance = object_target_info.get('green_distance', 1.0)  # Default to max distance if not found
            target_angle = object_target_info.get('green_angle', 0.0)  # Default to center if not found
            
            # Normalize distances to [0,1] range (0=close, 1=far)
            object_distance = min(max(object_distance, 0.0), 1.0)
            target_distance = min(max(target_distance, 0.0), 1.0)
            
            # Normalize angles to [-1,1] range (-1=left, 0=center, 1=right)
            object_angle = min(max(object_angle, -1.0), 1.0)
            target_angle = min(max(target_angle, -1.0), 1.0)
            
            return [object_detected, object_distance, object_angle, 
                   target_detected, target_distance, target_angle]
            
        except Exception as e:
            print(f"Error getting object-target state: {e}")
            return [0.0, 1.0, 0.0, 0.0, 1.0, 0.0]  # Default safe values
    
    def _check_task_completion(self):
        """Check if Task 3 object pushing is completed
        
        Task 3 completion depends on the current phase:
        - Phase 0-2: Not yet ready for completion
        - Phase 3: Complete when green target disappears (robot pushed object onto target)
        
        Returns:
            bool: True if task is completed, False otherwise
        """
        try:
            # Only check for completion in Phase 3
            if self.current_phase_num != 3:
                return False
            
            camera_frame = self.robot.read_image_front()
            if camera_frame is None:
                return False
            
            # Get object-target relationship from vision
            object_target_info = self.vision_processor.get_object_target_info_dict(camera_frame)
            
            # PHASE 3 COMPLETION CRITERIA: Green target disappears
            # This indicates the robot has pushed the object onto the green target
            green_targets_found = object_target_info.get('green_targets_found', False)
            
            # Initialize green disappearance tracking if needed
            if not hasattr(self, 'green_disappearance_count'):
                self.green_disappearance_count = 0
                self.last_green_seen_step = 0
            
            # Track green target visibility
            if green_targets_found:
                # Green target is visible - reset disappearance counter
                self.green_disappearance_count = 0
                self.last_green_seen_step = self.step_count
            else:
                # Green target not visible - increment disappearance counter
                self.green_disappearance_count += 1
            
            # Task completion: Green has disappeared for enough consecutive steps
            # This indicates robot is on top of the target
            required_disappearance_steps = 5  # Need 5 consecutive steps without green
            
            if self.green_disappearance_count >= required_disappearance_steps:
                print(f"🎯 PHASE 3 Task completion detected!")
                print(f"   Green target disappeared for {self.green_disappearance_count} consecutive steps")
                print(f"   Last seen green at step {self.last_green_seen_step}, current step {self.step_count}")
                print(f"   Robot successfully pushed object onto target!")
                return True
            
            # Debug info for Phase 3
            if self.step_count % 10 == 0:
                print(f"  Phase 3 Debug - Green visible: {green_targets_found}, "
                      f"Disappearance count: {self.green_disappearance_count}/{required_disappearance_steps}")
            
            return False
            
        except Exception as e:
            print(f"Error checking task completion: {e}")
            return False
            
        except Exception as e:
            print(f"Error checking task completion: {e}")
            return False
    


class ObjectPushVisionProcessor:
    """Computer Vision system for object pushing task (Task 3)
    
    This class handles all computer vision tasks for Task 3:
    - Red object detection and localization (object to be pushed)
    - Green target area detection and localization (destination)
    - Spatial relationship analysis between object and target
    - Distance and angle estimation for both object and target
    - Noise filtering and robust detection algorithms
    """
    
    def __init__(self, debug_mode: bool = False):
        self.debug_mode = debug_mode
        
        # Red color range for object detection (HSV)
        # Optimized for detecting the red object to be pushed
        self.red_lower = np.array([0, 100, 100])     # Lower HSV threshold for red
        self.red_upper = np.array([10, 255, 255])    # Upper HSV threshold for red
        self.red_lower2 = np.array([170, 100, 100])  # Second red range (wraparound)
        self.red_upper2 = np.array([180, 255, 255])  # Second red range (wraparound)
        
        # Green color range for target area detection (HSV)
        # Optimized for detecting the green target zone
        self.green_lower = np.array([35, 50, 50])    # Lower HSV threshold for green
        self.green_upper = np.array([85, 255, 255])  # Upper HSV threshold for green
        
        # Detection parameters
        self.min_contour_area = 100      # Minimum pixels for valid detection
        self.max_detection_distance = 3.0 # Maximum detection range (meters)
        
        # Noise filtering
        self.gaussian_blur_kernel = (5, 5)
        self.morphology_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        
    def detect_objects_and_target(self, camera_frame):
        """Detect red object and green target area in camera frame
        
        Returns:
            red_objects: List of detected red objects with [distance, angle, confidence]
            green_targets: List of detected green target areas with [distance, angle, confidence]
            processed_frame: Debug frame showing detection results
        """
        if camera_frame is None:
            return [], [], None
            
        try:
            # Convert to HSV for better color detection
            hsv = cv2.cvtColor(camera_frame, cv2.COLOR_BGR2HSV)
            
            # Create masks for red object (handle wraparound)
            red_mask1 = cv2.inRange(hsv, self.red_lower, self.red_upper)
            red_mask2 = cv2.inRange(hsv, self.red_lower2, self.red_upper2)
            red_mask = cv2.bitwise_or(red_mask1, red_mask2)
            
            # Create mask for green target area
            green_mask = cv2.inRange(hsv, self.green_lower, self.green_upper)
            
            # Apply noise reduction to both masks
            red_mask = self._apply_noise_reduction(red_mask)
            green_mask = self._apply_noise_reduction(green_mask)
            
            # Detect red objects
            red_objects = self._detect_objects_from_mask(red_mask, camera_frame.shape, "red")
            
            # Detect green target areas
            green_targets = self._detect_objects_from_mask(green_mask, camera_frame.shape, "green")
            
            # Create debug frame if requested
            processed_frame = None
            if self.debug_mode:
                processed_frame = self._create_debug_frame(camera_frame, red_objects, green_targets, red_mask, green_mask)
                            
            return red_objects, green_targets, processed_frame
            
        except Exception as e:
            print(f"Error in object and target detection: {e}")
            return [], [], None
    
    def _apply_noise_reduction(self, mask):
        """Apply noise reduction to a binary mask"""
        # Gaussian blur for noise reduction
        mask = cv2.GaussianBlur(mask, self.gaussian_blur_kernel, 0)
        # Morphological operations to fill gaps and remove noise
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.morphology_kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.morphology_kernel)
        return mask
    
    def _detect_objects_from_mask(self, mask, frame_shape, object_type):
        """Detect objects from a binary mask"""
        try:
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            detected_objects = []
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > self.min_contour_area:
                    # Calculate object properties
                    object_info = self._analyze_object_contour(contour, frame_shape, object_type)
                    if object_info:
                        detected_objects.append(object_info)
                        
            return detected_objects
            
        except Exception as e:
            print(f"Error detecting {object_type} objects: {e}")
            return []
    
    def _analyze_object_contour(self, contour, frame_shape, object_type):
        """Analyze object contour to extract distance and angle information"""
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
            if object_type == "red":
                # Red objects are typically smaller and closer
                distance = max(0.1, min(3.0, 1200.0 / (area + 1)))  # Adjusted for object size
            else:  # green target area
                # Green target areas are typically larger and may be farther
                distance = max(0.2, min(3.0, 2000.0 / (area + 1)))  # Adjusted for target size
            
            # Calculate confidence based on area and shape
            confidence = min(1.0, area / 1500.0)
            
            # Calculate shape metrics for validation
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
            else:
                circularity = 0
            
            return {
                'distance': distance,
                'angle': angle_degrees,
                'confidence': confidence,
                'centroid': (cx, cy),
                'area': area,
                'circularity': circularity,
                'type': object_type
            }
            
        except Exception as e:
            print(f"Error analyzing {object_type} contour: {e}")
            return None
    
    def _create_debug_frame(self, camera_frame, red_objects, green_targets, red_mask, green_mask):
        """Create debug visualization frame"""
        try:
            debug_frame = camera_frame.copy()
            
            # Draw red objects
            for obj in red_objects:
                centroid = obj['centroid']
                cv2.circle(debug_frame, centroid, 10, (0, 0, 255), 2)  # Red circle
                cv2.putText(debug_frame, f"R:{obj['distance']:.2f}m", 
                           (centroid[0] + 15, centroid[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            # Draw green targets
            for target in green_targets:
                centroid = target['centroid']
                cv2.circle(debug_frame, centroid, 15, (0, 255, 0), 2)  # Green circle
                cv2.putText(debug_frame, f"G:{target['distance']:.2f}m", 
                           (centroid[0] + 15, centroid[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            return debug_frame
            
        except Exception as e:
            print(f"Error creating debug frame: {e}")
            return camera_frame
    
    def get_object_target_relationship(self, camera_frame):
        """Get spatial relationship between object and target for state representation
        
        Returns: [object_detected, object_distance, object_angle, target_detected, target_distance, target_angle]
        """
        try:
            red_objects, green_targets, _ = self.detect_objects_and_target(camera_frame)
            
            # Default values: no detection
            object_detected = 0.0
            object_distance = 1.0
            object_angle = 0.0
            target_detected = 0.0
            target_distance = 1.0
            target_angle = 0.0
            
            # Find closest red object
            if red_objects:
                closest_object = min(red_objects, key=lambda obj: obj['distance'])
                object_detected = 1.0
                object_distance = min(closest_object['distance'] / self.max_detection_distance, 1.0)
                object_angle = closest_object['angle'] / 30.0  # Normalize angle to [-1, 1]
            
            # Find closest green target
            if green_targets:
                closest_target = min(green_targets, key=lambda target: target['distance'])
                target_detected = 1.0
                target_distance = min(closest_target['distance'] / self.max_detection_distance, 1.0)
                target_angle = closest_target['angle'] / 30.0  # Normalize angle to [-1, 1]
            
            return [object_detected, object_distance, object_angle, target_detected, target_distance, target_angle]
            
        except Exception as e:
            print(f"Error getting object-target relationship: {e}")
            return [0.0, 1.0, 0.0, 0.0, 1.0, 0.0]  # Default safe values
        
    def get_object_target_info_dict(self, camera_frame):
        """Get object and target information as dictionary for environment interface
        
        This method provides the expected interface for the environment to get
        vision information in dictionary format.
        
        Args:
            camera_frame: Input camera frame
            
        Returns:
            dict: Dictionary containing object and target detection information
        """
        try:
            red_objects, green_targets, _ = self.detect_objects_and_target(camera_frame)
            
            # Initialize default values
            result = {
                'red_objects_found': False,
                'red_distance': 1.0,  # Max distance when not found
                'red_angle': 0.0,     # Center when not found
                'red_center_x': 0.5,  # Center of frame
                'red_center_y': 0.5,  # Center of frame
                'red_size': 0.0,      # No size when not found
                'green_targets_found': False,
                'green_distance': 1.0,  # Max distance when not found
                'green_angle': 0.0,     # Center when not found
                'green_center_x': 0.5,  # Center of frame
                'green_center_y': 0.5,  # Center of frame
                'green_size': 0.0,      # No size when not found
            }
            
            # Process red objects (find closest)
            if red_objects:
                closest_red = min(red_objects, key=lambda obj: obj['distance'])
                result['red_objects_found'] = True
                result['red_distance'] = min(closest_red['distance'] / self.max_detection_distance, 1.0)
                result['red_angle'] = closest_red['angle'] / 30.0  # Normalize to [-1, 1]
                
                # Convert centroid to normalized coordinates
                if camera_frame is not None:
                    height, width = camera_frame.shape[:2]
                    result['red_center_x'] = closest_red['centroid'][0] / width
                    result['red_center_y'] = closest_red['centroid'][1] / height
                    result['red_size'] = min(closest_red['area'] / (width * height), 1.0)
            
            # Process green targets (find closest)
            if green_targets:
                closest_green = min(green_targets, key=lambda target: target['distance'])
                result['green_targets_found'] = True
                result['green_distance'] = min(closest_green['distance'] / self.max_detection_distance, 1.0)
                result['green_angle'] = closest_green['angle'] / 30.0  # Normalize to [-1, 1]
                
                # Convert centroid to normalized coordinates
                if camera_frame is not None:
                    height, width = camera_frame.shape[:2]
                    result['green_center_x'] = closest_green['centroid'][0] / width
                    result['green_center_y'] = closest_green['centroid'][1] / height
                    result['green_size'] = min(closest_green['area'] / (width * height), 1.0)
            
            return result
            
        except Exception as e:
            print(f"Error in get_object_target_info_dict: {e}")
            # Return safe default values on error
            return {
                'red_objects_found': False,
                'red_distance': 1.0,
                'red_angle': 0.0,
                'red_center_x': 0.5,
                'red_center_y': 0.5,
                'red_size': 0.0,
                'green_targets_found': False,
                'green_distance': 1.0,
                'green_angle': 0.0,
                'green_center_x': 0.5,
                'green_center_y': 0.5,
                'green_size': 0.0,
            }

    def _ir_intensity_to_distance(self, intensity_value: float) -> float:
        """Convert IR intensity value to distance in millimeters.
        
        Based on the Lua script calculations:
        - valorIR = a * (distancia[i] ^ b) where a = 0.1288, b = -1.7887
        - We need to reverse this: distance = (intensity / a) ^ (1/b)
        """
        if intensity_value <= 0:
            return 1000.0  # Maximum distance for no/invalid reading
        
        # Reverse the Lua calculation: distance = (intensity / a) ^ (1/b)
        a = 0.1288
        b = -1.7887
        
        try:
            # Calculate distance from intensity
            distance = (intensity_value / a) ** (1.0 / b)
            
            # Clamp to reasonable range (1mm to 1000mm) - allow very close distances
            distance = max(1.0, min(distance, 1000.0))
            return distance
        except (ZeroDivisionError, ValueError, OverflowError):
            return 1000.0  # Return max distance on calculation error

    # ...existing code...
def object_pushing_task3(rob: IRobobo, agent_type: str = 'dqn', mode: str = 'train', 
                         num_episodes: int = 100, collision_threshold: float = 0.95,
                         model_path: Optional[str] = None, max_episode_time: int = 300,
                         learning_rate: float = 0.001, gamma: float = 0.99,
                         epsilon: float = 1.0, epsilon_decay: float = 0.995,
                         epsilon_min: float = 0.01, save_freq: int = 100,
                         load_model_path: Optional[str] = None, 
                         save_model_path: Optional[str] = None, **kwargs):
    """Main function for Task 3: Object Pushing to Target using RL + Computer Vision
    
    This function orchestrates the complete object pushing task including:
    - Environment setup with computer vision for object and target detection
    - RL agent initialization and training
    - Performance evaluation and metrics collection
    - Model saving and visualization
    
    Args:
        rob: Robot interface (SimulationRobobo or HardwareRobobo)
        agent_type: RL algorithm ('dqn', 'qlearning', 'policy_gradient', 'actor_critic')
        mode: 'train', 'evaluate', or 'train_and_evaluate'
        num_episodes: Number of episodes to run
        collision_threshold: IR sensor threshold for collision detection
        model_path: Path to pre-trained model (legacy parameter, use load_model_path)
        max_episode_time: Maximum episode duration in seconds
        learning_rate: Learning rate for RL agent
        gamma: Discount factor for RL agent
        epsilon: Initial exploration rate for epsilon-greedy policies
        epsilon_decay: Decay rate for epsilon
        epsilon_min: Minimum epsilon value
        save_freq: Frequency for saving models (episodes)
        load_model_path: Path to load pre-trained model from
        save_model_path: Path to save trained model to
        **kwargs: Additional arguments
        
    Returns:
        Dict with training results, performance metrics, and saved file paths
    """
    
    print(f"\n🎯 TASK 3: OBJECT PUSHING TO TARGET")
    print(f"Agent: {agent_type.upper()}, Mode: {mode}, Episodes: {num_episodes}")
    print(f"Environment: {'Simulation' if isinstance(rob, SimulationRobobo) else 'Hardware'}")
    
    # Initialize computer vision processor for object and target detection
    vision_processor = ObjectPushVisionProcessor(debug_mode=True)
    
    # Create environment with optional image saving
    env = RobotEnvironment(rob, vision_processor, max_episode_time=max_episode_time, 
                          collision_threshold=collision_threshold, save_images=True, 
                          image_save_interval=100)  # Save every 100 steps
    
    # Get default hyperparameters and override with provided values
    hyperparams = get_default_hyperparameters(agent_type)
    hyperparams.update({
        'learning_rate': learning_rate,
        'gamma': gamma,
        'epsilon': epsilon,
        'epsilon_decay': epsilon_decay,
        'epsilon_min': epsilon_min
    })
    
    # Create RL agent
    agent = create_rl_agent(
        agent_type=agent_type,
        state_size=env.state_size,
        action_size=env.get_action_space_size(),
        **hyperparams
    )
    
    # Load pre-trained model if provided
    model_to_load = load_model_path or model_path  # Support both new and legacy parameter
    if model_to_load:
        print(f"Loading pre-trained model: {model_to_load}")
        try:
            agent.load_model(model_to_load)
            print("✅ Model loaded successfully!")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            if mode in ['evaluate', 'test']:
                return None
    
    # Performance tracking
    results = {
        'agent_type': agent_type,
        'mode': mode,
        'episodes': num_episodes,
        'episode_rewards': [],
        'episode_task_completions': [],
        'episode_lengths': [],
        'episode_success_rates': [],
        'training_time': 0,
        'best_episode_reward': float('-inf'),
        'best_task_completion': False,
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
                    task_completed = info.get('task_completed', False)
                    time_elapsed = time.time() - episode_start_time
                    action_name = env.action_descriptions[action]
                    print(f"  Step {step_count:3d} | Task: {'✅' if env.task_completed else '❌'} | "
                          f"Time: {time_elapsed:.1f}s | Reward: {total_reward:.1f} | "
                          f"Action: {action_name}")
            
            # Episode complete
            episode_time = time.time() - episode_start_time
            task_completed = env.task_completed
            success = task_completed
            
            # Record results
            results['episode_rewards'].append(total_reward)
            results['episode_task_completions'].append(1.0 if task_completed else 0.0)
            results['episode_lengths'].append(step_count)
            results['episode_success_rates'].append(1.0 if success else 0.0)
            
            # Track best performance
            if total_reward > results['best_episode_reward']:
                results['best_episode_reward'] = total_reward
            if task_completed and not results['best_task_completion']:
                results['best_task_completion'] = True
            
            # Episode summary
            success_rate = np.mean(results['episode_success_rates'][-10:]) * 100  # Last 10 episodes
            print(f"  ✅ Episode {episode + 1} Complete:")
            print(f"     Task Status: {'COMPLETED ✅' if task_completed else 'INCOMPLETE ❌'}")
            print(f"     Episode Time: {episode_time:.1f}s")
            print(f"     Episode Reward: {total_reward:.1f}")
            print(f"     Success Rate: {success_rate:.1f}%")
            
            # Save model periodically
            if (episode + 1) % save_freq == 0:
                timestamp = int(time.time())
                if save_model_path:
                    model_save_path = Path(save_model_path)
                    # Ensure we have a .pth extension
                    if not model_save_path.suffix:
                        model_save_path = model_save_path.with_suffix('.pth')
                else:
                    model_filename = f"rl_model_{agent_type}_{timestamp}.pth"
                    model_save_path = FIGURES_DIR / model_filename
                agent.save_model(str(model_save_path))
                print(f"     💾 Model saved: {model_save_path.name}")
        
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
            
            while not done and step_count < 3000:  # 5 minutes at 10 Hz
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
                    task_completed = info.get('task_completed', False)
                    time_elapsed = time.time() - episode_start_time
                    action_name = env.action_descriptions[action]
                    print(f"  Step {step_count:3d} | Task: {'✅' if env.task_completed else '❌'} | "
                          f"Time: {time_elapsed:.1f}s | Reward: {total_reward:.1f} | "
                          f"Action: {action_name}")
            
            # Episode complete
            episode_time = time.time() - episode_start_time
            task_completed = env.task_completed
            success = task_completed
            
            # Record results
            results['episode_rewards'].append(total_reward)
            results['episode_task_completions'].append(1.0 if task_completed else 0.0)
            results['episode_lengths'].append(step_count)
            results['episode_success_rates'].append(1.0 if success else 0.0)
            
            # Track best performance
            if total_reward > results['best_episode_reward']:
                results['best_episode_reward'] = total_reward
            if task_completed and not results['best_task_completion']:
                results['best_task_completion'] = True
            
            # Episode summary
            success_rate = np.mean(results['episode_success_rates'][-10:]) * 100  # Last 10 episodes
            print(f"  ✅ Evaluation Episode {episode + 1} Complete:")
            print(f"     Task Status: {'COMPLETED ✅' if task_completed else 'INCOMPLETE ❌'}")
            print(f"     Episode Time: {episode_time:.1f}s")
            print(f"     Episode Reward: {total_reward:.1f}")
            print(f"     Success Rate: {success_rate:.1f}%")
        

        
        results['evaluation_time'] = time.time() - start_time
        print(f"\n🎯 Evaluation completed in {results['evaluation_time']:.1f} seconds")
    
    # Save final model and results
    timestamp = int(time.time())
    
    # Save model
    if mode in ['train', 'train_and_evaluate']:
        if save_model_path:
            model_save_path = Path(save_model_path)
            # Ensure we have a .pth extension
            if not model_save_path.suffix:
                model_save_path = model_save_path.with_suffix('.pth')
        else:
            model_filename = f"rl_model_{agent_type}_{timestamp}.pth"
            model_save_path = FIGURES_DIR / model_filename
        
        agent.save_model(str(model_save_path))
        results['model_path'] = str(model_save_path)
        print(f"✅ Final model saved: {model_save_path.name}")
    
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
        avg_task_completions = np.mean(results['episode_task_completions'])
        final_success_rate = np.mean(results['episode_success_rates']) * 100
        
        print(f"\n📊 FINAL RESULTS:")
        print(f"   Average Reward: {avg_reward:.1f}")
        print(f"   Average Task Completions: {avg_task_completions:.1f}")
        print(f"   Success Rate: {final_success_rate:.1f}%")
        print(f"   Best Task Completion: {results['best_task_completion']}")
    
    return results



def test_task3_capabilities(rob: IRobobo):
    """Test all capabilities required for Task 3: Object Pushing to Target
    
    This function tests various components needed for Task 3 including:
    - Computer vision for object and target detection
    - Movement capabilities for navigation
    - Sensor readings for obstacle avoidance
    - Object manipulation mechanism
    
    Args:
        rob: Robot interface (SimulationRobobo or HardwareRobobo)
    """
    import time
    import numpy as np
    
    print("\n===== TESTING TASK 3 CAPABILITIES =====")
    
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
    
    # Test camera for object and target detection
    print("\n3. Testing camera for object and target detection...")
    image = rob.read_image_front()
    if image is not None:
        # Initialize object push vision processor
        vision_processor = ObjectPushVisionProcessor(debug_mode=True)
        object_target_info = vision_processor.get_object_target_relationship(image)
        
        print(f"Red objects detected: {object_target_info['red_objects_found']}")
        print(f"Green targets detected: {object_target_info['green_targets_found']}")
        
        if object_target_info['red_objects_found']:
            print(f"Red object center: ({object_target_info['red_center_x']:.2f}, {object_target_info['red_center_y']:.2f})")
            print(f"Red object size: {object_target_info['red_size']:.2f}")
        
        if object_target_info['green_targets_found']:
            print(f"Green target center: ({object_target_info['green_center_x']:.2f}, {object_target_info['green_center_y']:.2f})")
            print(f"Green target size: {object_target_info['green_size']:.2f}")
        
        # Save the debug image for inspection
        if isinstance(rob, SimulationRobobo):
            cv2.imwrite("results/figures/test_task3_camera.png", image)
    else:
        print("Failed to capture image from camera")
    
    # Test complete environment integration
    print("\n4. Testing environment integration...")
    try:
        vision_processor = ObjectPushVisionProcessor(debug_mode=True)
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
    
    print("\n===== TASK 3 CAPABILITY TESTING COMPLETE =====")


def demo_task3_object_pushing(rob: IRobobo, duration: int = 60):
    """Demo for Task 3: Object Pushing without RL
    
    This function demonstrates a simple object pushing behavior
    without using reinforcement learning. It uses basic computer vision
    and hardcoded behaviors to push red objects towards green targets.
    
    Args:
        rob: Robot interface (SimulationRobobo or HardwareRobobo)
        duration: Duration of the demo in seconds
    """
    from datetime import datetime
    
    print("\n===== TASK 3 OBJECT PUSHING DEMO =====")
    print(f"Running demo for {duration} seconds...")
    
    # Initialize vision processor
    vision_processor = ObjectPushVisionProcessor(debug_mode=True)
    
    # Reset robot state
    rob.set_emotion(Emotion.NORMAL)
    rob.move(0, 0)
    
    start_time = datetime.now()
    task_completions = 0
    
    while (datetime.now() - start_time).total_seconds() < duration:
        # Get camera image
        image = rob.read_image_front()
        
        if image is None:
            print("Failed to capture image, skipping frame")
            continue
        
        # Detect objects and targets
        object_target_info = vision_processor.get_object_target_relationship(image)
        
        # Simple behavior: move towards red object to push it towards green target
        if object_target_info['red_objects_found']:
            # Red object detected, move towards it
            x_center = object_target_info['red_center_x']
            y_center = object_target_info['red_center_y']
            size = object_target_info['red_size']
            
            # Check IR sensors for collision avoidance
            ir_values = rob.read_irs()
            front_collision = any(ir > 0.95 for ir in ir_values[:3])  # Front sensors
            
            if front_collision:
                # Object is close, check if we need to push in direction of target
                if object_target_info['green_targets_found']:
                    green_x = object_target_info['green_center_x']
                    # Push towards green target direction
                    if green_x < x_center:
                        rob.move(5, 15)  # Push left
                    elif green_x > x_center:
                        rob.move(15, 5)  # Push right
                    else:
                        rob.move(15, 15)  # Push forward
                else:
                    rob.move(10, 10)  # Just push forward
                rob.set_emotion(Emotion.HAPPY)
                print(f"Pushing red object towards target!")
            else:
                # Move towards red object
                if x_center < 0.4:
                    rob.move(5, 15)  # Turn left
                elif x_center > 0.6:
                    rob.move(15, 5)  # Turn right
                else:
                    rob.move(15, 15)  # Move forward
                rob.set_emotion(Emotion.NORMAL)
                print(f"Moving towards red object at x={x_center:.2f}, y={y_center:.2f}")
        
        elif object_target_info['green_targets_found']:
            # Only green target visible, search for red objects
            rob.move(-5, 5)  # Search by turning
            rob.set_emotion(Emotion.NORMAL)
            print("Target visible, searching for objects to push...")
            
        else:
            # No objects or targets detected, search by turning
            rob.move(-5, 5)  # Turn in place
            rob.set_emotion(Emotion.NORMAL)
            print("Searching for objects and targets...")
        
        # Check for task completion (when object is near target)
        if (object_target_info['red_objects_found'] and 
            object_target_info['green_targets_found']):
            red_x = object_target_info['red_center_x']
            green_x = object_target_info['green_center_x']
            distance = abs(red_x - green_x)
            
            if distance < 0.1:  # Object very close to target
                task_completions += 1
                rob.set_emotion(Emotion.LAUGHING)
                rob.play_emotion_sound(SoundEmotion.LAUGHING)
                print(f"Task completion detected! Total: {task_completions}")
        
        time.sleep(0.1)  # Small delay between iterations
    
    # Stop the robot at the end
    rob.move(0, 0)
    
    print(f"\nDemo completed! Task completions: {task_completions}")
    print("===== DEMO ENDED =====")


def test_object_vision_system(rob: IRobobo):
    """Test the computer vision system for object and target detection
    
    This function tests the ObjectPushVisionProcessor to ensure it can properly
    detect red objects and green targets in the camera feed.
    
    Args:
        rob: Robot interface (SimulationRobobo or HardwareRobobo)
    """
    print("\n===== TESTING OBJECT VISION SYSTEM =====")
    
    # Initialize vision processor
    vision_processor = ObjectPushVisionProcessor(debug_mode=True)
    
    print("Capturing camera images and testing detection...")
    
    for i in range(10):
        print(f"\nFrame {i+1}/10:")
        
        # Get camera image
        image = rob.read_image_front()
        
        if image is None:
            print("Failed to capture image")
            continue
        
        # Test object and target detection
        object_target_info = vision_processor.get_object_target_relationship(image)
        
        print(f"  Red objects: {object_target_info['red_objects_found']}")
        print(f"  Green targets: {object_target_info['green_targets_found']}")
        
        if object_target_info['red_objects_found']:
            print(f"  Red object at ({object_target_info['red_center_x']:.2f}, {object_target_info['red_center_y']:.2f})")
            print(f"  Red object size: {object_target_info['red_size']:.3f}")
        
        if object_target_info['green_targets_found']:
            print(f"  Green target at ({object_target_info['green_center_x']:.2f}, {object_target_info['green_center_y']:.2f})")
            print(f"  Green target size: {object_target_info['green_size']:.3f}")
        
        if object_target_info['red_objects_found'] and object_target_info['green_targets_found']:
            distance = abs(object_target_info['red_center_x'] - object_target_info['green_center_x'])
            print(f"  Object-target distance: {distance:.3f}")
        
        # Save debug images
        if isinstance(rob, SimulationRobobo):
            filename = f"results/figures/vision_test_frame_{i+1}.png"
            cv2.imwrite(filename, image)
        
        # Small rotation to get different views
        rob.move(-5, 5)
        time.sleep(0.5)
        rob.move(0, 0)
        time.sleep(0.5)
    
    rob.move(0, 0)  # Stop
    print("\n===== VISION SYSTEM TESTING COMPLETE =====")


def plot_task3_training_progress(results, agent_type='dqn'):
    """Plot and visualize training progress for Task 3 (Object Pushing)
    
    This function creates visualizations of the training progress including:
    - Episode rewards
    - Moving average rewards
    - Task completion statistics
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
    
    # Call the internal plotting function with Task 3 naming
    return _plot_task3_training_results(results, agent_type, timestamp)


def _plot_training_results(results, agent_type, timestamp):
    """Plot and save training results from RL runs"""
    from pathlib import Path
    
    # Create plots directory if it doesn't exist
    plots_dir = Path("results/figures")
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate plot file name with timestamp
    file_name = f"task3_{agent_type}_training_{timestamp}.png"
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
    
    # Plot task completion stats if available
    if 'episode_task_completions' in results:
        plt.subplot(2, 2, 3)
        plt.plot(episodes, results['episode_task_completions'], 'g-')
        plt.title('Task Completions')
        plt.xlabel('Episode')
        plt.ylabel('Number of Completions')
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


def _plot_task3_training_results(results, agent_type, timestamp):
    """Plot and save Task 3 training results from RL runs"""
    # Use the same plotting function but ensure Task 3 naming
    return _plot_training_results(results, agent_type, timestamp)
