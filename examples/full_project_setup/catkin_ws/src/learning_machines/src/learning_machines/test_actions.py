import cv2
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from data_files import FIGURES_DIR
from robobo_interface import (
    IRobobo,
    Emotion,
    LedId,
    LedColor,
    SoundEmotion,
    SimulationRobobo,
    HardwareRobobo,
)


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


def obstacle_avoidance(rob: IRobobo, iterations: int = 100, save_data: bool = True):
    """
    Simple obstacle avoidance behavior for Task 0.
    The robot moves forward until it detects an obstacle, then turns right.
    Handles differences between simulation and hardware IR sensors.
    
    Args:
        rob: Robot interface instance
        iterations: Number of control iterations to run
        save_data: Whether to save sensor and action data
    
    Returns:
        Collected data if save_data is True, otherwise None
    """
    # Prepare data collection
    data = {
        'time': [],
        'ir_sensors': [],
        'left_speed': [],
        'right_speed': []
    }
    
    if isinstance(rob, SimulationRobobo):
        rob.play_simulation()
    
    start_time = time.time()
    
    # Set default speeds
    forward_speed = 40
    turn_speed = 30
    
    # Obstacle avoidance threshold - different for simulation vs hardware
    if isinstance(rob, SimulationRobobo):
        threshold = 0.1  # Simulation threshold (lower values mean closer obstacles)
    else:
        threshold = 20  # Hardware threshold (lower values = closer obstacles)
    
    # Main control loop
    for i in range(iterations):
        # Read IR sensors
        ir_values = rob.read_irs()
        print(f"Iteration {i}: IR values: {ir_values}")
        
        # Front sensors are at indices 2, 3, 4 (FrontL, FrontR, FrontC)
        # Handle None values which might occur in hardware
        front_sensors = []
        for idx in [2, 3, 4]:  # FrontL, FrontC, FrontR
            if ir_values[idx] is not None:
                front_sensors.append(ir_values[idx])
            else:
                # If sensor reading is None, assume no obstacle (high value)
                front_sensors.append(1.0)
        
        # Collect data
        if save_data:
            current_time = time.time() - start_time
            data['time'].append(current_time)
            data['ir_sensors'].append(ir_values)
        
        # Check for obstacles with more robust detection
        obstacle_detected = False
        
        # Different detection logic based on platform
        if isinstance(rob, SimulationRobobo):
            # For simulation: simple threshold check
            obstacle_detected = any(sensor < threshold for sensor in front_sensors)
        else:
            # For hardware: more robust detection with multiple readings
            # Consider an obstacle detected if any sensor is below threshold
            # and handle potential sensor noise
            obstacle_detected = any(sensor < threshold for sensor in front_sensors)
            
            # If values are very close to threshold, take additional readings
            if not obstacle_detected and any(threshold <= sensor < threshold + 0.1 for sensor in front_sensors):
                # Take additional reading to confirm
                time.sleep(0.05)
                confirm_values = rob.read_irs()
                confirm_front = [confirm_values[2], confirm_values[3], confirm_values[4]]
                confirm_front = [s if s is not None else 1.0 for s in confirm_front]
                obstacle_detected = any(sensor < threshold for sensor in confirm_front)
        
        print(f"Obstacle detected: {obstacle_detected}")
        
        if obstacle_detected:
            # Obstacle detected - turn right and then continue forward
            print("Obstacle detected! Turning right.")
            try:
                rob.set_emotion(Emotion.SURPRISED)
            except:
                pass
                
            # First stop
            rob.move_blocking(0, 0, 100)
            
            # Turn right (90 degrees approximately)
            rob.move_blocking(turn_speed, -turn_speed, 1000)  # Turn for 1 second
            
            # Resume forward movement
            left_speed = forward_speed
            right_speed = forward_speed
        else:
            # No obstacle - continue moving forward
            left_speed = forward_speed
            right_speed = forward_speed
            try:
                rob.set_emotion(Emotion.HAPPY)
            except:
                pass
            print("Moving forward")
        
        # Execute movement
        rob.move_blocking(left_speed, right_speed, 200)  # 200ms per control cycle
        
        # Save action data
        if save_data:
            data['left_speed'].append(left_speed)
            data['right_speed'].append(right_speed)
    
    # Stop the robot
    rob.move_blocking(0, 0, 100)
    
    if isinstance(rob, SimulationRobobo):
        rob.stop_simulation()
    
    # Save data to CSV
    if save_data:
        # Convert to DataFrame
        df = pd.DataFrame({
            'time': data['time'],
            'left_speed': data['left_speed'],
            'right_speed': data['right_speed']
        })
        
        # Add IR sensor columns
        for i in range(8):
            df[f'ir_sensor_{i}'] = [sensors[i] if sensors[i] is not None else float('nan') 
                                  for sensors in data['ir_sensors']]
        
        # Save to CSV
        platform = 'simulation' if isinstance(rob, SimulationRobobo) else 'hardware'
        filename = f"task0_{platform}_data_{int(time.time())}.csv"
        csv_path = FIGURES_DIR / filename
        df.to_csv(csv_path, index=False)
        print(f"Data saved to {csv_path}")
        
        # Generate plots
        plot_sensor_data(df, platform)
        
        return data
    
    return None


def walk_until_obstacle(rob: IRobobo, duration_seconds: int = 60, save_data: bool = True):
    """
    Simple continuous walking behavior for Task 0.
    The robot moves forward until it detects an obstacle, then turns right.
    It continues this behavior for the specified duration.
    
    Args:
        rob: Robot interface instance
        duration_seconds: Duration to run the behavior in seconds
        save_data: Whether to save sensor and action data
    
    Returns:
        Collected data if save_data is True, otherwise None
    """
    # Prepare data collection
    data = {
        'time': [],
        'ir_sensors': [],
        'left_speed': [],
        'right_speed': [],
        'state': []  # 'forward' or 'turning'
    }
    
    if isinstance(rob, SimulationRobobo):
        rob.play_simulation()
    
    start_time = time.time()
    end_time = start_time + duration_seconds
    
    # Set default speeds
    forward_speed = 40
    turn_speed = 30
    
    # Obstacle avoidance threshold - different for simulation vs hardware
    if isinstance(rob, SimulationRobobo):
        threshold = 0.1  # Simulation threshold (lower values mean closer obstacles)
    else:
        threshold = 20  # Hardware threshold (higher values mean closer obstacles)
    
    # Initial state
    state = 'forward'
    left_speed = forward_speed
    right_speed = forward_speed
    
    # Main control loop
    while time.time() < end_time:
        current_time = time.time() - start_time
        
        # Read IR sensors
        ir_values = rob.read_irs()
        print(f"Time {current_time:.2f}s: IR values: {ir_values}")
        
        # Front sensors are at indices 2, 3, 4 (FrontL, FrontR, FrontC)
        # Handle None values which might occur in hardware
        front_sensors = []
        for idx in [2, 3, 4]:  # FrontL, FrontC, FrontR
            if ir_values[idx] is not None:
                front_sensors.append(ir_values[idx])
            else:
                # If sensor reading is None, assume no obstacle (high value)
                front_sensors.append(1.0)
        
        # Collect data
        if save_data:
            data['time'].append(current_time)
            data['ir_sensors'].append(ir_values)
            data['state'].append(state)
        
        # Check for obstacles
        if state == 'forward':
            # Different detection logic based on platform
            obstacle_detected = False
            if isinstance(rob, SimulationRobobo):
                # For simulation: simple threshold check
                obstacle_detected = any(sensor < threshold for sensor in front_sensors)
            else:
                # For hardware: more robust detection
                obstacle_detected = any(sensor < threshold for sensor in front_sensors)
            
            if obstacle_detected:
                # Obstacle detected - switch to turning state
                print("Obstacle detected! Turning right.")
                try:
                    rob.set_emotion(Emotion.SURPRISED)
                except:
                    pass
                
                # First stop
                rob.move_blocking(0, 0, 100)
                
                # Set turning speeds
                left_speed = turn_speed
                right_speed = -turn_speed
                state = 'turning'
                turn_start_time = time.time()
            else:
                # No obstacle - continue forward
                try:
                    rob.set_emotion(Emotion.HAPPY)
                except:
                    pass
        elif state == 'turning':
            # Check if we've turned for 1 second
            if time.time() - turn_start_time >= 1.0:
                # Resume forward movement
                state = 'forward'
                left_speed = forward_speed
                right_speed = forward_speed
        
        # Execute movement
        rob.move_blocking(left_speed, right_speed, 200)  # 200ms per control cycle
        
        # Save action data
        if save_data:
            data['left_speed'].append(left_speed)
            data['right_speed'].append(right_speed)
    
    # Stop the robot
    rob.move_blocking(0, 0, 100)
    
    if isinstance(rob, SimulationRobobo):
        rob.stop_simulation()
    
    # Save data to CSV
    if save_data:
        # Convert to DataFrame
        df = pd.DataFrame({
            'time': data['time'],
            'left_speed': data['left_speed'],
            'right_speed': data['right_speed'],
            'state': data['state']
        })
        
        # Add IR sensor columns
        for i in range(8):
            df[f'ir_sensor_{i}'] = [sensors[i] if sensors[i] is not None else float('nan') 
                                  for sensors in data['ir_sensors']]
        
        # Save to CSV
        platform = 'simulation' if isinstance(rob, SimulationRobobo) else 'hardware'
        filename = f"task0_{platform}_walk_data_{int(time.time())}.csv"
        csv_path = FIGURES_DIR / filename
        df.to_csv(csv_path, index=False)
        print(f"Data saved to {csv_path}")
        
        # Generate plots
        plot_sensor_data(df, platform)
        
        return data
    
    return None


def plot_sensor_data(df, platform='simulation'):
    """
    Generate plots for the collected sensor data
    
    Args:
        df: DataFrame with collected data
        platform: 'simulation' or 'hardware'
    """
    # Create a figure for all IR sensors
    plt.figure(figsize=(12, 8))
    
    # Plot IR sensor data over time
    for i in range(8):
        plt.plot(df['time'], df[f'ir_sensor_{i}'], label=f'IR Sensor {i}')
    
    plt.title(f'IR Sensor Readings over Time ({platform})')
    plt.xlabel('Time (s)')
    plt.ylabel('IR Sensor Value')
    plt.legend()
    plt.grid(True)
    
    # Save the figure
    plt.savefig(FIGURES_DIR / f'task0_{platform}_ir_sensors_{int(time.time())}.png')
    plt.close()
    
    # Create a figure for the front sensors only (more detailed analysis)
    plt.figure(figsize=(12, 8))
    
    # Plot only front IR sensors (2, 3, 4)
    for i in [2, 3, 4]:
        plt.plot(df['time'], df[f'ir_sensor_{i}'], label=f'Front IR Sensor {i}')
    
    plt.title(f'Front IR Sensor Readings over Time ({platform})')
    plt.xlabel('Time (s)')
    plt.ylabel('IR Sensor Value')
    plt.legend()
    plt.grid(True)
    
    # Save the figure
    plt.savefig(FIGURES_DIR / f'task0_{platform}_front_ir_sensors_{int(time.time())}.png')
    plt.close()
    
    # Create a figure for motor commands
    plt.figure(figsize=(12, 8))
    
    # Plot motor commands
    plt.plot(df['time'], df['left_speed'], label='Left Wheel Speed')
    plt.plot(df['time'], df['right_speed'], label='Right Wheel Speed')
    
    plt.title(f'Motor Commands over Time ({platform})')
    plt.xlabel('Time (s)')
    plt.ylabel('Motor Speed')
    plt.legend()
    plt.grid(True)
    
    # Save the figure
    plt.savefig(FIGURES_DIR / f'task0_{platform}_motor_commands_{int(time.time())}.png')
    plt.close()
    
    # If 'state' column exists (from walk_until_obstacle), plot state changes
    if 'state' in df.columns:
        plt.figure(figsize=(12, 8))
        
        # Create a numeric representation of state
        state_numeric = [1 if s == 'forward' else 0 for s in df['state']]
        
        # Plot state changes
        plt.plot(df['time'], state_numeric)
        plt.yticks([0, 1], ['turning', 'forward'])
        
        plt.title(f'Robot State over Time ({platform})')
        plt.xlabel('Time (s)')
        plt.ylabel('State')
        plt.grid(True)
        
        # Save the figure
        plt.savefig(FIGURES_DIR / f'task0_{platform}_state_{int(time.time())}.png')
        plt.close()
    
    print(f"Plots saved to {FIGURES_DIR}")


def run_all_actions(rob: IRobobo):
    if isinstance(rob, SimulationRobobo):
        rob.play_simulation()
    
    # Comment out test functions for Task 0
    # test_emotions(rob)
    # test_sensors(rob)
    # test_move_and_wheel_reset(rob)
    # if isinstance(rob, SimulationRobobo):
    #     test_sim(rob)
    # if isinstance(rob, HardwareRobobo):
    #     test_hardware(rob)
    # test_phone_movement(rob)
    
    # Run obstacle avoidance for Task 0
    obstacle_avoidance(rob, iterations=200, save_data=True)

    if isinstance(rob, SimulationRobobo):
        rob.stop_simulation()
