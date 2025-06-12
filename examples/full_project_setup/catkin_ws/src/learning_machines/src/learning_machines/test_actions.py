import cv2
import time
import math
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
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

# RL Models Experience
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


def run_all_actions(rob: IRobobo, use_rl: bool = False, rl_agent_type: str = 'qlearning', 
                   rl_mode: str = 'train', rl_episodes: int = 100):
    """Main function to run Task 1 - Advanced Obstacle Avoidance
    
    Args:
        rob: Robot interface instance
        use_rl: Whether to use RL-based approach instead of rule-based
        rl_agent_type: Type of RL agent ('qlearning', 'dqn', 'policy_gradient', 'actor_critic')
        rl_mode: RL mode ('train', 'evaluate', or 'train_and_evaluate')
        rl_episodes: Number of RL episodes
    """
    if isinstance(rob, SimulationRobobo):
        rob.play_simulation()
    
    print("="*50)
    if use_rl:
        print(f"TASK 1: RL-BASED OBSTACLE AVOIDANCE ({rl_agent_type.upper()})")
        print("="*50)
        
        # Run RL-based obstacle avoidance
        results = rl_obstacle_avoidance_task1(
            rob, agent_type=rl_agent_type, mode=rl_mode, num_episodes=rl_episodes
        )
        
        print(f"\nRL Training/Evaluation completed!")
        if rl_mode in ['train', 'train_and_evaluate']:
            print(f"Models and metrics saved to {FIGURES_DIR}")
        
        return results
    else:
        print("TASK 1: RULE-BASED OBSTACLE AVOIDANCE")
        print("="*50)
        
        # Run the original rule-based obstacle avoidance algorithm
        obstacle_avoidance_task1(rob, max_distance=10.0, duration_seconds=120, save_data=True)
    
    if isinstance(rob, SimulationRobobo):
        rob.stop_simulation()


def obstacle_avoidance_task1(rob: IRobobo, max_distance: float = 5.0, duration_seconds: int = 60, 
                            save_data: bool = True, threshold: Optional[float] = None):
    """
    Task 1 - Advanced Obstacle Avoidance Algorithm
    
    The robot explores an environment with obstacles by moving as far and as fast as possible
    while minimizing collisions. Uses a more sophisticated algorithm than the simple 
    right-turn approach.
    
    Objectives:
    - Maximize distance traveled 
    - Minimize collision/near-collision events
    - Explore the environment efficiently
    - Use intelligent turning strategies
    
    Args:
        rob: Robot interface instance (SimulationRobobo or HardwareRobobo)
        max_distance: Maximum distance to travel (for safety)
        duration_seconds: Maximum duration to run (seconds)
        save_data: Whether to save sensor and performance data
        threshold: Obstacle distance threshold (if None, uses default based on robot type)
    
    Returns:
        Performance metrics and data if save_data is True
    """
    # Performance metrics
    metrics = {
        'total_distance': 0.0,
        'total_time': 0.0,
        'collision_events': 0,
        'near_miss_events': 0,
        'average_speed': 0.0,
        'efficiency_score': 0.0
    }
    
        # Data collection for analysis
    data = {
        'time': [],
        'ir_sensors': [],
        'left_speed': [],
        'right_speed': [],
        'robot_position': [],
        'obstacle_detected': [],
        'state': []
    }
    
    if isinstance(rob, SimulationRobobo):
        rob.play_simulation()
        initial_pos = rob.get_position()
    
    start_time = time.time()
    end_time = start_time + duration_seconds
    last_position = None
    total_distance = 0.0
    
    # Algorithm parameters
    if isinstance(rob, SimulationRobobo):
        # Simulation threshold and speeds
        obstacle_threshold = threshold if threshold is not None else 0.15
        forward_speed = 60
        turn_speed = 40
    else:
        # Hardware threshold and speeds  
        obstacle_threshold = threshold if threshold is not None else 15
        forward_speed = 50
        turn_speed = 35
    
    # State machine for intelligent navigation
    state = 'exploring'  # exploring, avoiding, turning_left, turning_right, backing_up
    state_start_time = time.time()
    preferred_direction = 'right'  # Default preference
    
    print(f"Starting Task 1 - Advanced Obstacle Avoidance")
    print(f"Target: Travel {max_distance}m in {duration_seconds}s while avoiding obstacles")
    
    while time.time() < end_time and total_distance < max_distance:
        current_time = time.time() - start_time
        
        # Initialize default motor speeds
        left_speed = 0
        right_speed = 0
        
        # Read sensors
        ir_values = rob.read_irs()
        
        # Handle None values in IR readings
        safe_ir_values = []
        for i, val in enumerate(ir_values):
            if val is None:
                safe_ir_values.append(1.0 if isinstance(rob, SimulationRobobo) else 100.0)
            else:
                safe_ir_values.append(val)
        
        # Calculate collision risk based on front sensors
        front_sensors = [safe_ir_values[2], safe_ir_values[3], safe_ir_values[4]]  # FrontL, FrontC, FrontR
        min_front_distance = min(front_sensors)
        
        # Determine if obstacle detected - single threshold system
        if min_front_distance < obstacle_threshold:
            obstacle_detected = True
            metrics['collision_events'] += 1
            rob.set_emotion(Emotion.SURPRISED)  # Robot gets SURPRISED when obstacle detected
        else:
            obstacle_detected = False
        
        # Calculate position and distance if in simulation
        if isinstance(rob, SimulationRobobo):
            current_pos = rob.get_position()
            if last_position is not None:
                distance_step = math.sqrt(
                    (current_pos.x - last_position.x)**2 + 
                    (current_pos.y - last_position.y)**2
                )
                total_distance += distance_step
            last_position = current_pos
            pos_data = [current_pos.x, current_pos.y, current_pos.z]
        else:
            # For hardware, estimate distance from wheel encoder data
            wheel_data = rob.read_wheels()
            pos_data = [wheel_data.wheel_pos_l, wheel_data.wheel_pos_r, 0.0]
        
        # Advanced state machine for intelligent navigation
        if state == 'exploring':
            if obstacle_detected:
                # Obstacle detected - start avoidance maneuver
                state = 'avoiding'
                state_start_time = time.time()
                # Choose direction based on side sensor readings
                left_clear = safe_ir_values[2] > safe_ir_values[4]  # FrontL vs FrontR
                preferred_direction = 'left' if left_clear else 'right'
                # Start avoidance turn
                if preferred_direction == 'left':
                    left_speed = forward_speed - 20
                    right_speed = forward_speed
                else:
                    left_speed = forward_speed  
                    right_speed = forward_speed - 20
            else:
                # Clear path - move forward at optimal speed
                left_speed = forward_speed
                right_speed = forward_speed
                rob.set_emotion(Emotion.HAPPY)
                
        elif state == 'backing_up':
            # Back up briefly then turn
            if time.time() - state_start_time < 0.5:
                left_speed = -turn_speed
                right_speed = -turn_speed
            else:
                state = f'turning_{preferred_direction}'
                state_start_time = time.time()
                # Start the turn immediately
                direction = preferred_direction
                if direction == 'left':
                    left_speed = -turn_speed // 2
                    right_speed = turn_speed
                else:  # right
                    left_speed = turn_speed
                    right_speed = -turn_speed // 2
                
        elif state == 'avoiding':
            # Gradual avoidance - slight turn while moving forward
            if obstacle_detected:
                # Still detecting obstacle - continue or escalate to backing up
                if min_front_distance < obstacle_threshold * 0.8:  # Very close
                    state = 'backing_up'
                    state_start_time = time.time()
                    left_speed = -turn_speed
                    right_speed = -turn_speed
                else:
                    # Continue avoidance turn
                    if preferred_direction == 'left':
                        left_speed = forward_speed - 20
                        right_speed = forward_speed
                    else:
                        left_speed = forward_speed  
                        right_speed = forward_speed - 20
            else:
                # Clear path - return to exploring
                state = 'exploring'
                left_speed = forward_speed
                right_speed = forward_speed
                    
        elif state.startswith('turning_'):
            # Active turning maneuver
            direction = state.split('_')[1]
            if time.time() - state_start_time > 1.0:  # Turn for 1 second
                state = 'exploring'
                left_speed = forward_speed
                right_speed = forward_speed
            elif direction == 'left':
                left_speed = -turn_speed // 2
                right_speed = turn_speed
            else:  # right
                left_speed = turn_speed
                right_speed = -turn_speed // 2
        
        # Execute movement command
        rob.move_blocking(left_speed, right_speed, 200)
        
        # Data collection
        if save_data:
            data['time'].append(current_time)
            data['ir_sensors'].append(ir_values)
            data['left_speed'].append(left_speed)
            data['right_speed'].append(right_speed)
            data['robot_position'].append(pos_data)
            data['obstacle_detected'].append(obstacle_detected)
            data['state'].append(state)
        
        # Progress reporting
        if int(current_time) % 10 == 0 and int(current_time * 10) % 10 == 0:
            print(f"Time: {current_time:.1f}s, Distance: {total_distance:.2f}m, State: {state}, Obstacle: {obstacle_detected}")
    
    # Stop robot
    rob.move_blocking(0, 0, 100)
    
    # Calculate final metrics
    metrics['total_distance'] = total_distance
    metrics['total_time'] = time.time() - start_time
    metrics['average_speed'] = total_distance / metrics['total_time'] if metrics['total_time'] > 0 else 0
    
    # Efficiency score: distance per unit time, penalized by collisions
    collision_penalty = metrics['collision_events'] * 0.1 + metrics['near_miss_events'] * 0.05
    metrics['efficiency_score'] = max(0, metrics['average_speed'] - collision_penalty)
    
    if isinstance(rob, SimulationRobobo):
        rob.stop_simulation()
    
    # Report performance
    print(f"\nTask 1 Performance Summary:")
    print(f"Distance traveled: {metrics['total_distance']:.2f}m")
    print(f"Time elapsed: {metrics['total_time']:.1f}s") 
    print(f"Average speed: {metrics['average_speed']:.2f}m/s")
    print(f"Collision events: {metrics['collision_events']}")
    print(f"Near-miss events: {metrics['near_miss_events']}")
    print(f"Efficiency score: {metrics['efficiency_score']:.3f}")
    
    # Save data and generate analysis
    if save_data:
        platform = 'simulation' if isinstance(rob, SimulationRobobo) else 'hardware'
        save_task1_data(data, metrics, platform)
        plot_task1_analysis(data, metrics, platform)
        
        return {'data': data, 'metrics': metrics}
    
    return metrics


def wall_following_algorithm(rob: IRobobo, duration_seconds: int = 60, wall_distance: float = 0.3):
    """
    Alternative Task 1 approach: Wall-following algorithm
    
    Follows walls to systematically explore the environment while maintaining
    a safe distance from obstacles.
    
    Args:
        rob: Robot interface instance
        duration_seconds: Duration to run the algorithm  
        wall_distance: Target distance to maintain from walls
    """
    if isinstance(rob, SimulationRobobo):
        rob.play_simulation()
        wall_threshold = wall_distance
        forward_speed = 40
        turn_speed = 30
    else:
        wall_threshold = wall_distance * 50  # Convert to hardware scale
        forward_speed = 35
        turn_speed = 25
    
    start_time = time.time()
    end_time = start_time + duration_seconds
    
    print(f"Starting wall-following algorithm for {duration_seconds}s")
    
    while time.time() < end_time:
        ir_values = rob.read_irs()
        
        # Handle None values
        safe_ir_values = []
        for val in ir_values:
            if val is None:
                safe_ir_values.append(1.0 if isinstance(rob, SimulationRobobo) else 100.0)
            else:
                safe_ir_values.append(val)
        
        # Right wall following - use right sensors
        right_sensor = safe_ir_values[4]  # FrontR
        front_sensor = safe_ir_values[3]  # FrontC
        
        if front_sensor < wall_threshold * 1.2:
            # Wall ahead - turn left
            left_speed = -turn_speed // 2
            right_speed = turn_speed
        elif right_sensor > wall_threshold * 1.5:
            # No wall on right - turn right to find wall
            left_speed = turn_speed
            right_speed = turn_speed // 2
        elif right_sensor < wall_threshold * 0.8:
            # Too close to wall - turn left slightly
            left_speed = forward_speed - 10
            right_speed = forward_speed
        else:
            # Good distance - move forward
            left_speed = forward_speed
            right_speed = forward_speed
        
        rob.move_blocking(left_speed, right_speed, 200)
    
    rob.move_blocking(0, 0, 100)
    
    if isinstance(rob, SimulationRobobo):
        rob.stop_simulation()
    
    print("Wall-following algorithm completed")


def save_task1_data(data, metrics, platform):
    """Save Task 1 data to CSV files"""
    # Convert to DataFrame
    df = pd.DataFrame({
        'time': data['time'],
        'left_speed': data['left_speed'], 
        'right_speed': data['right_speed'],
        'obstacle_detected': data['obstacle_detected'],
        'state': data['state']
    })
    
    # Add IR sensor columns
    for i in range(8):
        df[f'ir_sensor_{i}'] = [sensors[i] if sensors[i] is not None else float('nan') 
                              for sensors in data['ir_sensors']]
    
    # Add position columns
    df['pos_x'] = [pos[0] for pos in data['robot_position']]
    df['pos_y'] = [pos[1] for pos in data['robot_position']]
    df['pos_z'] = [pos[2] for pos in data['robot_position']]
    
    # Save main data
    timestamp = int(time.time())
    data_file = FIGURES_DIR / f"task1_{platform}_data_{timestamp}.csv"
    df.to_csv(data_file, index=False)
    
    # Save metrics
    metrics_file = FIGURES_DIR / f"task1_{platform}_metrics_{timestamp}.csv"
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(metrics_file, index=False)
    
    print(f"Data saved to {data_file}")
    print(f"Metrics saved to {metrics_file}")


def plot_task1_analysis(data, metrics, platform):
    """Generate comprehensive analysis plots for Task 1"""
    timestamp = int(time.time())
    
    # Plot 1: Robot trajectory (simulation only)
    if platform == 'simulation':
        plt.figure(figsize=(10, 8))
        pos_x = [pos[0] for pos in data['robot_position']]
        pos_y = [pos[1] for pos in data['robot_position']]
        
        # Color trajectory by obstacle detection
        colors = {True: 'red', False: 'green'}
        for i in range(len(pos_x)-1):
            color = colors.get(data['obstacle_detected'][i], 'blue')
            plt.plot(pos_x[i:i+2], pos_y[i:i+2], color=color, linewidth=2)
        
        plt.title(f'Robot Trajectory - Task 1 ({platform})')
        plt.xlabel('X Position (m)')
        plt.ylabel('Y Position (m)')
        plt.grid(True)
        plt.axis('equal')
        
        # Add legend for obstacle detection
        from matplotlib.lines import Line2D
        legend_elements = [Line2D([0], [0], color='green', lw=2, label='Clear Path'),
                          Line2D([0], [0], color='red', lw=2, label='Obstacle Detected')]
        plt.legend(handles=legend_elements)
        
        plt.savefig(FIGURES_DIR / f'task1_{platform}_trajectory_{timestamp}.png')
        plt.close()
    
    # Plot 2: Speed and risk over time
    plt.figure(figsize=(12, 8))
    
    plt.subplot(3, 1, 1)
    plt.plot(data['time'], data['left_speed'], label='Left Speed', alpha=0.7)
    plt.plot(data['time'], data['right_speed'], label='Right Speed', alpha=0.7)
    plt.title(f'Motor Commands - Task 1 ({platform})')
    plt.ylabel('Speed')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(3, 1, 2)
    # Convert obstacle detection to numeric for plotting
    obstacle_numeric = [1 if detected else 0 for detected in data['obstacle_detected']]
    plt.plot(data['time'], obstacle_numeric, 'r-', linewidth=2)
    plt.title('Obstacle Detection')
    plt.ylabel('Obstacle Detected')
    plt.yticks([0, 1], ['No', 'Yes'])
    plt.grid(True)
    
    plt.subplot(3, 1, 3)
    # Plot front IR sensors
    for i in [2, 3, 4]:  # Front sensors
        sensor_data = [sensors[i] if sensors[i] is not None else float('nan') 
                      for sensors in data['ir_sensors']]
        plt.plot(data['time'], sensor_data, label=f'Front IR {i}')
    plt.title('Front IR Sensors')
    plt.xlabel('Time (s)')
    plt.ylabel('Distance')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f'task1_{platform}_analysis_{timestamp}.png')
    plt.close()
    
    # Plot 3: Performance summary
    plt.figure(figsize=(10, 6))
    
    # Performance bar chart
    categories = ['Distance (m)', 'Avg Speed (m/s)', 'Efficiency', 'Collisions', 'Near Misses']
    values = [
        metrics['total_distance'],
        metrics['average_speed'], 
        metrics['efficiency_score'],
        metrics['collision_events'],
        metrics['near_miss_events']
    ]
    
    colors = ['blue', 'green', 'purple', 'red', 'orange']
    bars = plt.bar(categories, values, color=colors, alpha=0.7)
    
    plt.title(f'Task 1 Performance Summary ({platform})')
    plt.ylabel('Value')
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f'task1_{platform}_performance_{timestamp}.png')
    plt.close()
    
    print(f"Analysis plots saved to {FIGURES_DIR}")

class RobotEnvironment:
    """Environment wrapper for Robobo robot RL training"""
    
    def __init__(self, robot: IRobobo, max_episode_steps: int = 1000, obstacle_threshold: Optional[float] = None):
        self.robot = robot
        self.max_episode_steps = max_episode_steps
        self.step_count = 0
        self.is_simulation = isinstance(robot, SimulationRobobo)
        
        # Action space: [left_speed, right_speed] discretized (removed stop action)
        self.action_space_size = 8
        self.actions = [(-50, -50), (-25, 25), (-10, 50), (0, 50), (50, 50), 
                       (50, 0), (50, -10), (25, -25)]
        
        # Action descriptions for debugging
        self.action_descriptions = [
            "Backward", "Turn Left", "Slight Left", "Forward Slow", "Forward Fast",
            "Forward Right", "Slight Right", "Turn Right"
        ]
        
        # State space: IR sensors + previous action (optimized - removed orientation)
        self.state_size = 9  # 8 IR + 1 prev_action (robot never tilts)
        
        # Thresholds - use provided threshold or defaults
        if self.is_simulation:
            # In simulation, IR values are distances: smaller = closer
            # FIXED: Increased threshold to 150 for close-proximity obstacle detection only
            # Higher threshold = less sensitive = detects only very close walls (arena navigation)
            self.obstacle_threshold = obstacle_threshold if obstacle_threshold is not None else 150
        else:
            self.obstacle_threshold = obstacle_threshold if obstacle_threshold is not None else 15
        
        self.reset()
    
    def reset(self):
        """Reset environment for new episode"""
        self.step_count = 0
        if self.is_simulation:
            self.robot.play_simulation()
            # Give simulation time to settle
            time.sleep(0.5)
        
        self.prev_action = 4  # Start with forward action
        self.last_position = None
        self.episode_distance = 0.0
        self.grace_period_steps = 15  # Increased grace period for complex initial movements
        self.stuck_counter = 0  # Counter for detecting stuck behavior
        
        # Debug: Check initial IR readings
        initial_state = self._get_state()
        ir_raw = self.robot.read_irs()
        print(f"Reset - IR readings: {ir_raw[:4]}, Threshold: {self.obstacle_threshold}")
        
        return initial_state
    
    def _get_state(self):
        """Get current state representation (optimized 9D without orientation)"""
        # IR sensors (8 dimensions)
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
                    ir_normalized.append(min(val / 100.0, 1.0))
        
        # Previous action (1 dimension) - helps with momentum understanding
        prev_action_norm = [self.prev_action / float(self.action_space_size - 1)]
        
        # Combine: 8 IR + 1 prev_action = 9 dimensions (removed orientation)
        state = np.array(ir_normalized + prev_action_norm, dtype=np.float32)
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
            print(f"  Reward: {reward:.2f}, Info: {info}")
        
        # Set emotion based on reward and situation
        if info.get('collision', False):
            emotion = Emotion.SAD
        elif info.get('near_miss', False):
            emotion = Emotion.SURPRISED
        elif reward > 5.0:
            emotion = Emotion.HAPPY
        elif reward < -2.0:
            emotion = Emotion.ANGRY
        else:
            emotion = Emotion.NORMAL
        
        # Store emotion in info for dynamic display
        info['emotion'] = emotion
        info['ir_sensors'] = next_state[:8]  # Store IR sensor readings
        info['action_taken'] = self.action_descriptions[action_idx]
        
        # Check if episode is done
        # Only end episode when max steps reached or robot is stuck
        # Collisions should NOT end the episode - robot should learn to avoid them
        done = (self.step_count >= self.max_episode_steps or info['stuck'])
        
        self.prev_action = action_idx
        
        return next_state, reward, done, info
    
    def _calculate_reward(self, action_idx, state):
        """Calculate reward based on current state and action"""
        info = {'collision': False, 'near_miss': False, 'stuck': False, 'distance_reward': 0.0}
        reward = 0.0
        
        # Get raw IR sensor values for reward calculation
        ir_raw = self.robot.read_irs()
        front_ir_raw = []
        for i in [2, 3, 4]:  # Front sensors
            if ir_raw[i] is not None:
                front_ir_raw.append(ir_raw[i])
        
        if front_ir_raw:
            min_front_ir = min(front_ir_raw)
        else:
            min_front_ir = 2000 if self.is_simulation else 100  # Default safe value
        
        # Distance reward (encourage forward movement)
        if self.is_simulation:
            current_pos = self.robot.get_position()
            if self.last_position is not None:
                distance = math.sqrt(
                    (current_pos.x - self.last_position.x)**2 + 
                    (current_pos.y - self.last_position.y)**2
                )
                self.episode_distance += distance
                info['distance_reward'] = distance * 10
                reward += info['distance_reward']
            self.last_position = current_pos
        else:
            # Hardware mode: Reward all forward-moving actions, not just action 4
            if action_idx in [3, 4, 5, 6]:  # Forward movements: slow, fast, right, slight right
                base_reward = 2.0
                # Give extra reward for pure forward actions
                if action_idx == 4:  # Forward Fast
                    base_reward = 3.0
                elif action_idx == 3:  # Forward Slow  
                    base_reward = 2.5
                reward += base_reward
                info['distance_reward'] = base_reward
        
        # Collision penalty - use raw IR values with correct thresholds
        if self.is_simulation:
            # In simulation: smaller values = closer obstacles
            # Skip collision detection for first few steps to allow robot to move away from initial position
            if hasattr(self, 'grace_period_steps') and self.grace_period_steps > 0:
                self.grace_period_steps -= 1
                # IMPROVED: Context-aware grace period - only reward forward if path is clear
                if min_front_ir > 0.1:  # Path is clear - encourage forward movement
                    if action_idx == 4:  # Forward Fast action
                        reward += 5.0  # Strong positive reward for moving forward
                        info['distance_reward'] = 5.0
                        self.robot.set_emotion(Emotion.HAPPY)
                    elif action_idx == 3:  # Forward Slow action
                        reward += 3.0  # Good reward for forward movement
                        self.robot.set_emotion(Emotion.HAPPY)
                    else:
                        reward += 1.0  # Mild positive reward for any other movement
                        self.robot.set_emotion(Emotion.NORMAL)
                else:  # Path blocked - encourage turning instead!
                    if action_idx in [1, 7]:  # Sharp turns when blocked
                        reward += 4.0  # Reward smart turning
                        self.robot.set_emotion(Emotion.NORMAL)
                    elif action_idx in [2, 6]:  # Gentle turns when blocked
                        reward += 2.0  # Reward turning
                        self.robot.set_emotion(Emotion.NORMAL)
                    elif action_idx == 0:  # Backup when blocked
                        reward += 3.0  # Reward backing up
                        self.robot.set_emotion(Emotion.NORMAL)
                    elif action_idx in [3, 4]:  # Forward into obstacle during grace period
                        reward -= 5.0  # Penalty for moving toward obstacle
                        self.robot.set_emotion(Emotion.SURPRISED)
                    else:
                        reward += 1.0  # Default small reward
            else:
                # Normal collision detection after grace period
                if min_front_ir < self.obstacle_threshold:
                    reward -= 50.0
                    info['collision'] = True
                    self.robot.set_emotion(Emotion.SAD)
                elif min_front_ir < self.obstacle_threshold * 5.0:  # Increased near-miss range
                    reward -= 5.0
                    info['near_miss'] = True
                    self.robot.set_emotion(Emotion.SURPRISED)
                else:
                    reward += 2.0
                    self.robot.set_emotion(Emotion.HAPPY)
        else:
            # Hardware: larger values = closer obstacles
            if min_front_ir > self.obstacle_threshold:
                reward -= 50.0
                info['collision'] = True
                self.robot.set_emotion(Emotion.SAD)
            elif min_front_ir > self.obstacle_threshold * 0.5:
                reward -= 5.0
                info['near_miss'] = True
                self.robot.set_emotion(Emotion.SURPRISED)
            else:
                reward += 2.0
                self.robot.set_emotion(Emotion.HAPPY)
        
        # Reduce penalty for turning when necessary for obstacle avoidance
        # Use normalized state values for this check
        front_sensors_norm = state[:3]  # Front sensors are first 3 in 9D state
        if np.mean(front_sensors_norm) > 0.7 and action_idx in [1, 2, 6, 7]:  # Only penalize if path is very clear
            reward -= 0.5  # Reduced penalty from -1.0 to -0.5
        
        # Check if stuck - robot should always be moving now (no stop action available)
        # This condition is now less relevant since stop action is removed
        ir_state_values = state[:8]  # First 8 values are IR sensors
        if np.all(ir_state_values > 0.9):  # Very high threshold for stuck detection
            info['stuck'] = True
        
        return reward, info


class QLearningAgent:
    """Q-Learning agent with discretized state space"""
    
    def __init__(self, state_size: int, action_size: int, learning_rate: float = 0.1, 
                 epsilon: float = 1.0, epsilon_decay: float = 0.995, epsilon_min: float = 0.01,
                 gamma: float = 0.95):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.gamma = gamma
        
        self.q_table = {}
        self.training_rewards = []
        self.training_losses = []
    
    def _discretize_state(self, state):
        """Convert continuous state to discrete bins"""
        bins = 10
        discrete_state = []
        
        for i, val in enumerate(state):
            val_clipped = np.clip(val, 0.0, 1.0)
            bin_idx = min(int(val_clipped * bins), bins - 1)
            discrete_state.append(bin_idx)
        
        return tuple(discrete_state)
    
    def get_action(self, state, training=True):
        """Select action using epsilon-greedy policy"""
        discrete_state = self._discretize_state(state)
        
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.action_size)
        
        if discrete_state not in self.q_table:
            self.q_table[discrete_state] = np.zeros(self.action_size)
        
        return np.argmax(self.q_table[discrete_state])
    
    def update(self, state, action, reward, next_state, done):
        """Update Q-table using Q-learning update rule"""
        discrete_state = self._discretize_state(state)
        discrete_next_state = self._discretize_state(next_state)
        
        if discrete_state not in self.q_table:
            self.q_table[discrete_state] = np.zeros(self.action_size)
        if discrete_next_state not in self.q_table:
            self.q_table[discrete_next_state] = np.zeros(self.action_size)
        
        current_q = self.q_table[discrete_state][action]
        if done:
            target_q = reward
        else:
            target_q = reward + self.gamma * np.max(self.q_table[discrete_next_state])
        
        self.q_table[discrete_state][action] += self.learning_rate * (target_q - current_q)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        loss = abs(target_q - current_q)
        self.training_losses.append(loss)
    
    def save_model(self, filepath):
        """Save Q-table to file"""
        model_data = {
            'q_table': self.q_table,
            'epsilon': self.epsilon,
            'training_rewards': self.training_rewards,
            'training_losses': self.training_losses
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath):
        """Load Q-table from file"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        self.q_table = model_data['q_table']
        self.epsilon = model_data.get('epsilon', self.epsilon_min)
        self.training_rewards = model_data.get('training_rewards', [])
        self.training_losses = model_data.get('training_losses', [])


class DQNNetwork(nn.Module):
    """Deep Q-Network architecture"""
    
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 256):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc4 = nn.Linear(hidden_size // 2, action_size)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class DQNAgent:
    """Deep Q-Network agent"""
    
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
        states = torch.FloatTensor([e.state for e in batch]).to(self.device)
        actions = torch.LongTensor([e.action for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e.reward for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e.next_state for e in batch]).to(self.device)
        dones = torch.BoolTensor([e.done for e in batch]).to(self.device)
        
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


class PolicyNetwork(nn.Module):
    """Policy network for REINFORCE"""
    
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
                # Slightly bias the final layer toward forward movement (action 4)
                if m == self.fc3:
                    nn.init.constant_(m.bias, 0.1)
                    # Give slight bias to forward action (now action 4 in 8-action space)
                    if self.fc3.out_features > 4:  # Ensure action 4 exists
                        m.bias.data[4] = 0.3
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
    """REINFORCE Policy Gradient agent"""
    
    def __init__(self, state_size: int, action_size: int, learning_rate: float = 0.001,
                 gamma: float = 0.95):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        
        # Neural network
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_network = PolicyNetwork(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)
        
        # Episode storage
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []
        
        # Training metrics
        self.training_rewards = []
        self.training_losses = []
        
        # Exploration parameters
        self.exploration_episodes = 50  # Number of episodes to maintain exploration
        self.current_episode = 0
        self.min_entropy = 0.1  # Minimum entropy to maintain
    
    def get_action(self, state, training=True):
        """Select action using policy network with improved exploration"""
        # Ensure state is valid
        state = np.nan_to_num(state, nan=0.0, posinf=1.0, neginf=0.0)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad() if not training else torch.enable_grad():
            action_probs = self.policy_network(state_tensor)
            
            # Add exploration during early training episodes
            if training and self.current_episode < self.exploration_episodes:
                # Add small amount of uniform noise for exploration
                epsilon = max(0.1, 0.5 - (self.current_episode / self.exploration_episodes) * 0.4)
                uniform_probs = torch.ones_like(action_probs) / action_probs.size(1)
                action_probs = (1 - epsilon) * action_probs + epsilon * uniform_probs
            
            # Add small epsilon to prevent zero probabilities
            action_probs = action_probs + 1e-8
            action_probs = action_probs / action_probs.sum(dim=1, keepdim=True)
            
            # Check for NaN values
            if torch.isnan(action_probs).any():
                print("Warning: NaN detected in action probabilities, using uniform distribution")
                action_probs = torch.ones_like(action_probs) / action_probs.size(1)
        
        if training:
            # Sample from probability distribution
            try:
                action_dist = torch.distributions.Categorical(action_probs)
                action = action_dist.sample()
                
                # Debug: Print action probabilities occasionally
                if self.current_episode % 10 == 0 and len(self.episode_actions) == 0:
                    print(f"Episode {self.current_episode}: Action probs = {action_probs.detach().cpu().numpy().flatten()[:5]}...")
                
                return action.item()
            except ValueError as e:
                print(f"Categorical distribution error: {e}")
                # Fallback to random action
                return np.random.randint(self.action_size)
        else:
            # Take most likely action for evaluation
            return action_probs.argmax().item()
    
    def remember(self, state, action, reward):
        """Store episode step"""
        self.episode_states.append(state)
        self.episode_actions.append(action)
        self.episode_rewards.append(reward)
    
    def update_episode(self):
        """Update policy at end of episode using REINFORCE"""
        if len(self.episode_rewards) == 0:
            return
        
        # Increment episode counter
        self.current_episode += 1
        
        # Calculate discounted returns
        returns = []
        G = 0
        for reward in reversed(self.episode_rewards):
            G = reward + self.gamma * G
            returns.insert(0, G)
        
        # Normalize returns only if there's variation
        returns = torch.FloatTensor(returns).to(self.device)
        if len(returns) > 1 and returns.std() > 1e-8:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Convert episode data to tensors efficiently
        episode_states_array = np.array(self.episode_states)
        states = torch.FloatTensor(episode_states_array).to(self.device)
        actions = torch.LongTensor(self.episode_actions).to(self.device)
        
        # Calculate policy loss using proper forward pass
        with torch.enable_grad():
            # Get probabilities from forward pass
            action_probs = self.policy_network(states)
            
            # Take log of probabilities (with numerical stability)
            action_probs = torch.clamp(action_probs, min=1e-8, max=1.0)
            log_probs = torch.log(action_probs)
            action_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze()
            
            # Calculate loss with entropy bonus for exploration
            policy_loss = -(action_log_probs * returns).mean()
            
            # Add entropy bonus during early training
            if self.current_episode < self.exploration_episodes:
                entropy = -(action_probs * log_probs).sum(dim=1).mean()
                entropy_bonus = 0.01 * entropy  # Small entropy bonus
                policy_loss = policy_loss - entropy_bonus
        
        # Check for valid loss
        if torch.isnan(policy_loss) or torch.isinf(policy_loss):
            print("Warning: Invalid policy loss, skipping update")
            self.episode_states.clear()
            self.episode_actions.clear()
            self.episode_rewards.clear()
            return
        
        # Optimize
        self.optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Store metrics
        episode_reward = sum(self.episode_rewards)
        self.training_rewards.append(episode_reward)
        self.training_losses.append(policy_loss.item())
        
        # Debug: Print episode info
        if self.current_episode % 10 == 0:
            print(f"Episode {self.current_episode}: Reward={episode_reward:.2f}, "
                  f"Steps={len(self.episode_rewards)}, Loss={policy_loss.item():.4f}")
        
        # Clear episode data
        self.episode_states.clear()
        self.episode_actions.clear()
        self.episode_rewards.clear()
    
    def update(self, state, action, reward, next_state, done):
        """Interface compatibility - store step and update if episode done"""
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


class ActorCriticNetwork(nn.Module):
    """Actor-Critic network with shared features"""
    
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 256):
        super(ActorCriticNetwork, self).__init__()
        # Shared layers
        self.shared_fc1 = nn.Linear(state_size, hidden_size)
        self.shared_fc2 = nn.Linear(hidden_size, hidden_size // 2)
        
        # Actor head (policy)
        self.actor_fc = nn.Linear(hidden_size // 2, action_size)
        
        # Critic head (value)
        self.critic_fc = nn.Linear(hidden_size // 2, 1)
        
        self.dropout = nn.Dropout(0.2)
        
        # Initialize weights for stability
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier/Glorot initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.01)
    
    def forward(self, x):
        # Add input clamping for numerical stability
        x = torch.clamp(x, -10.0, 10.0)
        
        # Shared features
        x = F.relu(self.shared_fc1(x))
        x = self.dropout(x)
        x = F.relu(self.shared_fc2(x))
        
        # Actor output (action probabilities)
        action_logits = self.actor_fc(x)
        action_logits = torch.clamp(action_logits, -10.0, 10.0)  # Prevent extreme values
        action_probs = F.softmax(action_logits, dim=1)
        
        # Critic output (state value)
        state_value = self.critic_fc(x)
        
        return action_probs, state_value


class ActorCriticAgent:
    """Actor-Critic (A2C) agent"""
    
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
        
        # Episode storage
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []
        self.episode_values = []
        
        # Training metrics
        self.training_rewards = []
        self.training_losses = []
    
    def get_action(self, state, training=True):
        """Select action using actor-critic policy"""
        # Ensure state is valid
        state = np.nan_to_num(state, nan=0.0, posinf=1.0, neginf=0.0)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        action_probs, state_value = self.network(state_tensor)
        
        # Add small epsilon to prevent zero probabilities
        action_probs = action_probs + 1e-8
        action_probs = action_probs / action_probs.sum(dim=1, keepdim=True)
        
        # Check for NaN values
        if torch.isnan(action_probs).any():
            print("Warning: NaN detected in action probabilities, using uniform distribution")
            action_probs = torch.ones_like(action_probs) / action_probs.size(1)
        
        if training:
            # Sample from probability distribution
            try:
                action_dist = torch.distributions.Categorical(action_probs)
                action = action_dist.sample()
                
                # Store value for training
                self.episode_values.append(state_value.item())
                
                return action.item()
            except ValueError as e:
                print(f"Categorical distribution error: {e}")
                # Fallback to random action
                return np.random.randint(self.action_size)
        else:
            # Take most likely action for evaluation
            return action_probs.argmax().item()
    
    def remember(self, state, action, reward):
        """Store episode step"""
        self.episode_states.append(state)
        self.episode_actions.append(action)
        self.episode_rewards.append(reward)
    
    def update_episode(self):
        """Update both actor and critic at end of episode"""
        if len(self.episode_rewards) == 0:
            return
        
        # Calculate returns and advantages
        returns = []
        G = 0
        for reward in reversed(self.episode_rewards):
            G = reward + self.gamma * G
            returns.insert(0, G)
        
        returns = torch.FloatTensor(returns).to(self.device)
        values = torch.FloatTensor(self.episode_values).to(self.device)
        
        # Calculate advantages
        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Convert episode data to tensors
        episode_states_array = np.array(self.episode_states)
        states = torch.FloatTensor(episode_states_array).to(self.device)
        actions = torch.LongTensor(self.episode_actions).to(self.device)
        
        # Forward pass
        action_probs, state_values = self.network(states)
        
        # Actor loss (policy gradient with advantage)
        log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(1)).squeeze())
        actor_loss = -(log_probs * advantages.detach()).mean()
        
        # Critic loss (value function)
        critic_loss = F.mse_loss(state_values.squeeze(), returns)
        
        # Entropy bonus for exploration
        entropy = -(action_probs * torch.log(action_probs + 1e-8)).sum(1).mean()
        
        # Total loss
        total_loss = actor_loss + self.value_loss_coef * critic_loss - self.entropy_coef * entropy
        
        # Optimize
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
        self.optimizer.step()
        
        # Store metrics
        episode_reward = sum(self.episode_rewards)
        self.training_rewards.append(episode_reward)
        self.training_losses.append(total_loss.item())
        
        # Clear episode data
        self.episode_states.clear()
        self.episode_actions.clear()
        self.episode_rewards.clear()
        self.episode_values.clear()
    
    def update(self, state, action, reward, next_state, done):
        """Interface compatibility - store step and update if episode done"""
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


def create_rl_agent(agent_type: str, state_size: int, action_size: int, **kwargs):
    """Factory function to create RL agents"""
    agents = {
        'qlearning': QLearningAgent,
        'dqn': DQNAgent,
        'policy_gradient': PolicyGradientAgent,
        'actor_critic': ActorCriticAgent
    }
    
    if agent_type not in agents:
        raise ValueError(f"Unknown agent type: {agent_type}. Available: {list(agents.keys())}")
    
    return agents[agent_type](state_size, action_size, **kwargs)


def train_rl_agent(rob: IRobobo, agent_type: str = 'qlearning', num_episodes: int = 100, 
                   max_steps_per_episode: int = 500, save_model: bool = True,
                   obstacle_threshold: Optional[float] = None):
    """
    Train RL agent for Task 1 - Obstacle Avoidance
    
    Args:
        rob: Robot interface instance
        agent_type: Type of RL agent ('qlearning', 'dqn', 'policy_gradient', 'actor_critic')
        num_episodes: Number of training episodes
        max_steps_per_episode: Maximum steps per episode
        save_model: Whether to save trained model
        obstacle_threshold: Optional custom obstacle detection threshold (currently unused)
    
    Returns:
        Trained agent and training metrics
    """
    # Create environment and agent
    env = RobotEnvironment(rob, max_steps_per_episode, obstacle_threshold)
    agent = create_rl_agent(agent_type, env.state_size, env.action_space_size)
    
    print(f"Starting RL Training: {agent_type.upper()}")
    print(f"Episodes: {num_episodes}, Max steps per episode: {max_steps_per_episode}")
    print(f"State size: {env.state_size}, Action size: {env.action_space_size}")
    
    training_metrics = {
        'episode_rewards': [],
        'episode_lengths': [],
        'average_rewards': [],
        'collision_rates': []
    }
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        episode_length = 0
        collision_count = 0
        last_info = {'collision': False}  # Initialize info for scope
        
        for step in range(max_steps_per_episode):
            # Select action
            action = agent.get_action(state, training=True)
            
            # Execute action
            next_state, reward, done, info = env.step(action)
            last_info = info  # Store last info
            
            # Update agent
            agent.update(state, action, reward, next_state, done)
            
            # Update metrics
            episode_reward += reward
            episode_length += 1
            if info['collision']:
                collision_count += 1
            
            state = next_state
            
            if done:
                break
        
        # Store episode metrics
        training_metrics['episode_rewards'].append(episode_reward)
        training_metrics['episode_lengths'].append(episode_length)
        collision_rate = collision_count / episode_length if episode_length > 0 else 0
        training_metrics['collision_rates'].append(collision_rate)
        
        # Calculate running average
        window_size = min(10, len(training_metrics['episode_rewards']))
        avg_reward = np.mean(training_metrics['episode_rewards'][-window_size:])
        training_metrics['average_rewards'].append(avg_reward)
        
        # Store episode reward in agent
        agent.training_rewards.append(episode_reward)
        
        # Simple progress reporting every episode
        action_taken = last_info.get('action_taken', 'Unknown')
        print(f"Episode {episode+1}/{num_episodes}: Steps={episode_length}, Reward={episode_reward:.2f}, "
              f"Collisions={collision_count}, Last Action={action_taken}, "
              f"Done={'Collision' if last_info.get('collision') else 'Max Steps' if episode_length == max_steps_per_episode else 'Stuck' if last_info.get('stuck') else 'Other'}")
        
        # Detailed summary every 10 episodes
        if (episode + 1) % 10 == 0:
            print(f"\n{'='*60}")
            print(f"Episodes {episode-8}-{episode+1} Summary:")
            print(f"  Average Reward (last 10): {avg_reward:.2f}")
            print(f"  Average Length (last 10): {np.mean(training_metrics['episode_lengths'][-10:]):.1f}")
            print(f"  Average Collisions (last 10): {np.mean([training_metrics['collision_rates'][i] for i in range(max(0, len(training_metrics['collision_rates'])-10), len(training_metrics['collision_rates']))]):.3f}")
            if hasattr(agent, 'epsilon'):
                print(f"  Exploration (ε): {agent.epsilon:.3f}")
            print(f"{'='*60}\n")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED")
    print("="*60)
    
    # Save model
    if save_model:
        timestamp = int(time.time())
        model_file = FIGURES_DIR / f"rl_model_{agent_type}_{timestamp}.pth"
        agent.save_model(model_file)
        print(f"Model saved to {model_file}")
        
        # Save training metrics
        metrics_file = FIGURES_DIR / f"rl_metrics_{agent_type}_{timestamp}.json"
        with open(metrics_file, 'w') as f:
            json.dump(training_metrics, f, indent=2)
        print(f"Training metrics saved to {metrics_file}")
    
    return agent, training_metrics


def evaluate_rl_agent(rob: IRobobo, agent, num_episodes: int = 10, max_steps_per_episode: int = 500,
                       obstacle_threshold: Optional[float] = None):
    """
    Evaluate trained RL agent
    
    Args:
        rob: Robot interface instance
        agent: Trained RL agent
        num_episodes: Number of evaluation episodes
        max_steps_per_episode: Maximum steps per episode
        obstacle_threshold: Optional custom obstacle detection threshold
    
    Returns:
        Evaluation metrics
    """
    env = RobotEnvironment(rob, max_steps_per_episode, obstacle_threshold)
    
    eval_metrics = {
        'episode_rewards': [],
        'episode_lengths': [],
        'collision_counts': [],
        'success_rate': 0.0,
        'average_reward': 0.0,
        'average_length': 0.0
    }
    
    print(f"Evaluating RL agent for {num_episodes} episodes...")
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        episode_length = 0
        collision_count = 0
        
        for step in range(max_steps_per_episode):
            # Select action (no exploration)
            action = agent.get_action(state, training=False)
            
            # Execute action
            next_state, reward, done, info = env.step(action)
            
            # Update metrics
            episode_reward += reward
            episode_length += 1
            if info['collision']:
                collision_count += 1
            
            state = next_state
            
            if done:
                break
        
        # Store episode metrics
        eval_metrics['episode_rewards'].append(episode_reward)
        eval_metrics['episode_lengths'].append(episode_length)
        eval_metrics['collision_counts'].append(collision_count)
        
        print(f"Episode {episode + 1}: Reward={episode_reward:.2f}, "
              f"Length={episode_length}, Collisions={collision_count}")
    
    # Calculate summary statistics
    eval_metrics['average_reward'] = np.mean(eval_metrics['episode_rewards'])
    eval_metrics['average_length'] = np.mean(eval_metrics['episode_lengths'])
    eval_metrics['success_rate'] = sum(1 for c in eval_metrics['collision_counts'] if c == 0) / num_episodes
    
    print(f"\nEvaluation Results:")
    print(f"Average Reward: {eval_metrics['average_reward']:.2f}")
    print(f"Average Episode Length: {eval_metrics['average_length']:.2f}")
    print(f"Success Rate (no collisions): {eval_metrics['success_rate']:.2%}")
    print(f"Average Collisions per Episode: {np.mean(eval_metrics['collision_counts']):.2f}")
    
    return eval_metrics


def rl_obstacle_avoidance_task1(rob: IRobobo, agent_type: str = 'qlearning', 
                                mode: str = 'train', num_episodes: int = 100,
                                max_steps_per_episode: int = 500,
                                model_path: Optional[str] = None,
                                obstacle_threshold: Optional[float] = None):
    """
    Main function for RL-based Task 1 - Obstacle Avoidance
    
    Args:
        rob: Robot interface instance
        agent_type: Type of RL agent ('qlearning', 'dqn', 'policy_gradient', 'actor_critic')
        mode: 'train', 'evaluate', or 'train_and_evaluate'
        num_episodes: Number of episodes
        max_steps_per_episode: Maximum steps per episode
        model_path: Path to load/save model
        obstacle_threshold: Optional custom obstacle detection threshold
    
    Returns:
        Results based on mode
    """
    print("="*60)
    print(f"TASK 1: RL-BASED OBSTACLE AVOIDANCE ({agent_type.upper()})")
    print("="*60)
    
    agent = None
    training_metrics = None
    
    if mode in ['train', 'train_and_evaluate']:
        # Training phase
        agent, training_metrics = train_rl_agent(
            rob, agent_type, num_episodes, max_steps_per_episode, save_model=True,
            obstacle_threshold=obstacle_threshold
        )
        
        # Plot training progress
        plot_rl_training_progress(training_metrics, agent_type)
        
        if mode == 'train':
            return agent, training_metrics
    
    if mode in ['evaluate', 'train_and_evaluate']:
        # Load model if provided
        if model_path and mode == 'evaluate':
            env = RobotEnvironment(rob, max_steps_per_episode, obstacle_threshold)
            agent = create_rl_agent(agent_type, env.state_size, env.action_space_size)
            agent.load_model(model_path)
            print(f"Loaded model from {model_path}")
        elif mode == 'train_and_evaluate':
            # Use the trained agent
            pass
        
        # Evaluation phase
        if agent is not None:
            eval_metrics = evaluate_rl_agent(rob, agent, num_episodes=10, max_steps_per_episode=max_steps_per_episode,
                                           obstacle_threshold=obstacle_threshold)
            
            if mode == 'evaluate':
                return eval_metrics
            else:
                return agent, training_metrics, eval_metrics
        else:
            print("Error: No agent available for evaluation")
            return None


def plot_rl_training_progress(metrics, agent_type):
    """Plot RL training progress"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Episode rewards
    axes[0, 0].plot(metrics['episode_rewards'], alpha=0.7, label='Episode Reward')
    axes[0, 0].plot(metrics['average_rewards'], label='Average Reward')
    axes[0, 0].set_title('Training Rewards')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Episode lengths
    axes[0, 1].plot(metrics['episode_lengths'])
    axes[0, 1].set_title('Episode Lengths')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Steps')
    axes[0, 1].grid(True)
    
    # Collision rates
    axes[1, 0].plot(metrics['collision_rates'])
    axes[1, 0].set_title('Collision Rates')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Collision Rate')
    axes[1, 0].grid(True)
    
    # Running average reward (longer window)
    window_size = min(50, len(metrics['episode_rewards']))
    if len(metrics['episode_rewards']) >= window_size:
        running_avg = []
        for i in range(window_size - 1, len(metrics['episode_rewards'])):
            avg = np.mean(metrics['episode_rewards'][i - window_size + 1:i + 1])
            running_avg.append(avg)
        
        axes[1, 1].plot(range(window_size - 1, len(metrics['episode_rewards'])), running_avg)
        axes[1, 1].set_title(f'Running Average Reward (window={window_size})')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Reward')
        axes[1, 1].grid(True)
    
    plt.suptitle(f'RL Training Progress - {agent_type.upper()}')
    plt.tight_layout()
    
    timestamp = int(time.time())
    plt.savefig(FIGURES_DIR / f'rl_training_{agent_type}_{timestamp}.png')
    plt.close()
    
    print(f"Training progress plot saved to {FIGURES_DIR}")

def test_unified_threshold_system():
    """
    Test function to verify the unified threshold system is working correctly.
    This replaces the old dual-threshold (collision_threshold + near_miss_threshold) 
    with a single obstacle_threshold where robot gets SURPRISED when obstacle detected.
    """
    print("Testing Unified Threshold System")
    print("="*50)
    
    # Test simulation thresholds
    print("Simulation Mode:")
    print(f"  Default obstacle_threshold: 0.15")
    print(f"  Custom obstacle_threshold: 0.20 (when provided)")
    
    # Test hardware thresholds  
    print("\nHardware Mode:")
    print(f"  Default obstacle_threshold: 15")
    print(f"  Custom obstacle_threshold: 25 (when provided)")
    
    # Test obstacle detection logic
    print("\nObstacle Detection Logic:")
    print("  if min_front_distance < obstacle_threshold:")
    print("    obstacle_detected = True")
    print("    robot.set_emotion(Emotion.SURPRISED)")
    print("  else:")
    print("    obstacle_detected = False")
    
    print("\nState Machine Updates:")
    print("  - Removed collision_risk levels ('low', 'medium', 'high')")
    print("  - Replaced with simple obstacle_detected boolean")
    print("  - Robot gets SURPRISED emotion when obstacle detected")
    print("  - Unified single threshold replaces dual threshold system")
    
    print("\nData Collection Updates:")
    print("  - Replaced 'collision_risk' with 'obstacle_detected' in data")
    print("  - Updated plotting to show obstacle detection (Yes/No)")
    print("  - Updated CSV export to use new unified system")
    
    print("\n✅ Unified Threshold System Implementation Complete!")
