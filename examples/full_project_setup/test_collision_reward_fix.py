#!/usr/bin/env python3
"""
Test script to verify the collision reward fix in the RL system.
This script tests that the reward function properly penalizes collisions
instead of giving positive rewards.
"""

import sys
import os
import numpy as np

# Add the source directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
catkin_src_path = os.path.join(current_dir, 'catkin_ws/src')
learning_machines_path = os.path.join(catkin_src_path, 'learning_machines/src')

sys.path.insert(0, learning_machines_path)
sys.path.insert(0, catkin_src_path)

try:
    from learning_machines.test_actions import RobotEnvironment
    from learning_machines.robobo_interface import Emotion
except ImportError as e:
    print(f"Import error: {e}")
    print(f"Current directory: {current_dir}")
    print(f"Looking for modules in: {learning_machines_path}")
    print("\nTrying alternative import...")
    
    # Alternative import method
    sys.path.insert(0, os.path.join(current_dir, 'catkin_ws/src/learning_machines/src/learning_machines'))
    from test_actions import RobotEnvironment
    from robobo_interface import Emotion

class MockRobot:
    """Mock robot for testing reward function"""
    
    def __init__(self, simulation=True):
        self.is_simulation = simulation
        self.emotion_set = None
        self._ir_values = None
        
    def read_irs(self):
        """Return mock IR sensor readings"""
        if self._ir_values is None:
            # Default safe readings
            if self.is_simulation:
                return [2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000]  # Far from obstacles
            else:
                return [0, 0, 0, 0, 0, 0, 0, 0]  # Far from obstacles
        return self._ir_values
    
    def set_ir_values(self, values):
        """Set mock IR sensor readings for testing"""
        self._ir_values = values
        
    def get_orientation(self):
        """Return mock orientation"""
        class MockOrientation:
            yaw = 0.0
        return MockOrientation()
        
    def move_blocking(self, left, right, millis):
        """Mock movement command"""
        pass
        
    def set_emotion(self, emotion):
        """Mock emotion setting"""
        self.emotion_set = emotion
        
    def play_simulation(self):
        """Mock simulation play"""
        pass

def test_collision_scenarios():
    """Test various collision scenarios to ensure negative rewards"""
    
    print("="*80)
    print("TESTING COLLISION REWARD FIX")
    print("="*80)
    
    # Test both simulation and hardware scenarios
    for is_simulation in [True, False]:
        platform = "Simulation" if is_simulation else "Hardware"
        print(f"\n--- Testing {platform} Platform ---")
        
        mock_robot = MockRobot(simulation=is_simulation)
        env = RobotEnvironment(mock_robot, max_episode_steps=100)
        
        # Set up collision thresholds
        if is_simulation:
            # Simulation: lower values = closer obstacles
            collision_values = [50, 100, 149]  # Values that should trigger collision (< 150)
            safe_values = [200, 500, 1000]     # Safe values (> 150)
        else:
            # Hardware: higher values = closer obstacles  
            collision_values = [20, 30, 50]    # Values that should trigger collision (> 15)
            safe_values = [5, 10, 14]          # Safe values (< 15)
        
        # Test 1: Forward movement into collision
        print(f"\n  Test 1: Forward movement collision detection")
        for collision_val in collision_values:
            # Set front sensors to collision values
            if is_simulation:
                ir_values = [2000, 2000, 2000, 2000, collision_val, collision_val, 2000, collision_val]
            else:
                ir_values = [0, 0, 0, 0, collision_val, collision_val, 0, collision_val]
            
            mock_robot.set_ir_values(ir_values)
            state = env._get_state()
            
            # Test forward action (action 4 or 5)
            reward, info = env._calculate_reward(4, state)  # Forward action
            
            collision_detected = info.get('collision', False)
            print(f"    IR front sensors: {collision_val}, Reward: {reward:.2f}, Collision: {collision_detected}")
            
            if collision_detected and reward >= 0:
                print(f"    ❌ ERROR: Positive reward ({reward:.2f}) given for collision!")
                return False
            elif collision_detected and reward < 0:
                print(f"    ✅ CORRECT: Negative reward ({reward:.2f}) for collision")
            elif not collision_detected:
                print(f"    ⚠️  WARNING: Collision not detected with value {collision_val}")
        
        # Test 2: Turning during grace period with collision
        print(f"\n  Test 2: Grace period turning collision detection")
        env.grace_period_steps = 10  # Set grace period
        
        for collision_val in collision_values:
            # Set sensors to collision values
            if is_simulation:
                ir_values = [collision_val, collision_val, 2000, 2000, collision_val, collision_val, collision_val, collision_val]
            else:
                ir_values = [collision_val, collision_val, 0, 0, collision_val, collision_val, collision_val, collision_val]
            
            mock_robot.set_ir_values(ir_values)
            state = env._get_state()
            
            # Test turning action during grace period (action 1 = turn)
            reward, info = env._calculate_reward(1, state)  # Turn action
            
            collision_detected = info.get('collision', False)
            print(f"    Grace period turn with collision IR: {collision_val}, Reward: {reward:.2f}, Collision: {collision_detected}")
            
            if collision_detected and reward >= 0:
                print(f"    ❌ ERROR: Positive reward ({reward:.2f}) given for grace period collision!")
                return False
            elif collision_detected and reward < 0:
                print(f"    ✅ CORRECT: Negative reward ({reward:.2f}) for grace period collision")
        
        # Test 3: Safe movements should get positive rewards
        print(f"\n  Test 3: Safe movement reward verification")
        for safe_val in safe_values:
            # Set all sensors to safe values
            if is_simulation:
                ir_values = [safe_val] * 8
            else:
                ir_values = [safe_val] * 8
            
            mock_robot.set_ir_values(ir_values)
            state = env._get_state()
            
            # Test forward action with safe values
            reward, info = env._calculate_reward(4, state)  # Forward action
            
            collision_detected = info.get('collision', False)
            near_miss_detected = info.get('near_miss', False)
            
            print(f"    Safe forward IR: {safe_val}, Reward: {reward:.2f}, Collision: {collision_detected}, Near-miss: {near_miss_detected}")
            
            if collision_detected:
                print(f"    ❌ ERROR: False collision detected with safe value {safe_val}")
                return False
            elif reward <= 0:
                print(f"    ⚠️  WARNING: Non-positive reward ({reward:.2f}) for safe movement")
            else:
                print(f"    ✅ CORRECT: Positive reward ({reward:.2f}) for safe movement")
    
    print(f"\n{'='*80}")
    print("✅ ALL COLLISION REWARD TESTS PASSED!")
    print("The reward function now properly penalizes collisions and does not give positive rewards for wall hits.")
    print("="*80)
    return True

def test_reward_consistency():
    """Test that rewards are consistent between platforms"""
    
    print(f"\n{'='*80}")
    print("TESTING REWARD CONSISTENCY BETWEEN PLATFORMS")
    print("="*80)
    
    # Create environments for both platforms
    sim_robot = MockRobot(simulation=True)
    hw_robot = MockRobot(simulation=False)
    
    sim_env = RobotEnvironment(sim_robot, max_episode_steps=100)
    hw_env = RobotEnvironment(hw_robot, max_episode_steps=100)
    
    # Test collision scenarios with equivalent sensor readings
    print("\nTesting equivalent collision scenarios:")
    
    # Simulation collision (< 150) vs Hardware collision (> 15)
    sim_robot.set_ir_values([2000, 2000, 2000, 2000, 100, 100, 2000, 100])  # Collision in sim
    hw_robot.set_ir_values([0, 0, 0, 0, 25, 25, 0, 25])  # Collision in hardware
    
    sim_state = sim_env._get_state()
    hw_state = hw_env._get_state()
    
    sim_reward, sim_info = sim_env._calculate_reward(4, sim_state)  # Forward action
    hw_reward, hw_info = hw_env._calculate_reward(4, hw_state)    # Forward action
    
    print(f"Simulation collision reward: {sim_reward:.2f}, Collision: {sim_info.get('collision', False)}")
    print(f"Hardware collision reward: {hw_reward:.2f}, Collision: {hw_info.get('collision', False)}")
    
    # Both should detect collisions and give negative rewards
    if (sim_info.get('collision', False) and hw_info.get('collision', False) and 
        sim_reward < 0 and hw_reward < 0):
        print("✅ CORRECT: Both platforms properly detect collisions with negative rewards")
        return True
    else:
        print("❌ ERROR: Inconsistent collision detection between platforms")
        return False

if __name__ == "__main__":
    try:
        # Run collision tests
        collision_test_passed = test_collision_scenarios()
        
        if collision_test_passed:
            # Run consistency tests
            consistency_test_passed = test_reward_consistency()
            
            if consistency_test_passed:
                print(f"\n🎉 ALL TESTS PASSED! The collision reward system is working correctly.")
                sys.exit(0)
            else:
                print(f"\n❌ Consistency tests failed!")
                sys.exit(1)
        else:
            print(f"\n❌ Collision tests failed!")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n❌ Test error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
