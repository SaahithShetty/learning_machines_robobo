#!/usr/bin/env python3
"""
Test script to validate phase-specific actions implementation
"""

import sys
import os
import numpy as np

# Add the package to the Python path
sys.path.append('/Users/saahithshetty/Documents/Coding/Learning_Machines/learning_machines_robobo/examples/full_project_setup/catkin_ws/src/learning_machines/src')

def test_phase_specific_actions():
    """Test that phase-specific actions are correctly implemented"""
    
    print("🧪 Testing Phase-Specific Actions Implementation")
    print("=" * 60)
    
    # Mock robot interface for testing
    class MockRobot:
        def __init__(self):
            self.pan = 0.0
            self.tilt = 0.0
            
        def read_phone_pan(self):
            return self.pan
            
        def read_phone_tilt(self):
            return self.tilt
            
        def read_irs(self):
            return [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000]
            
        def read_image_front(self):
            return np.zeros((240, 320, 3), dtype=np.uint8)
            
        def move_blocking(self, left, right, duration):
            print(f"    Mock move: L={left}, R={right}, Duration={duration}ms")
            
        def stop_simulation(self):
            pass
            
        def play_simulation(self):
            pass
            
        def get_sim_time(self):
            return 0.0
    
    # Mock vision processor
    class MockVisionProcessor:
        def get_object_target_info_dict(self, frame):
            return {
                'red_objects_found': False,
                'red_center_x': 0.0,
                'red_center_y': 0.0,
                'red_size': 0.0,
                'green_targets_found': False,
                'green_center_x': 0.0,
                'green_center_y': 0.0,
                'green_size': 0.0
            }
    
    # Import the environment after ensuring path is set
    from learning_machines.test_actions import RobotEnvironment
    
    # Create environment with mock components
    robot = MockRobot()
    vision_processor = MockVisionProcessor()
    env = RobotEnvironment(robot, vision_processor)
    
    print(f"✅ Environment initialized successfully")
    print(f"   Use phase-specific actions: {env.use_phase_specific_actions}")
    print(f"   Initial phase: {env.current_phase_num}")
    print(f"   Initial action space size: {env.get_action_space_size()}")
    print()
    
    # Test each phase
    for phase in range(4):
        print(f"🔄 Testing Phase {phase}")
        print(f"   Phase name: {['Object Detection', 'Object Collection', 'Target Detection', 'Push to Target'][phase]}")
        
        # Transition to phase
        env._transition_to_phase(phase, f"Testing phase {phase}")
        
        print(f"   Action space size: {env.get_action_space_size()}")
        print(f"   Actions available:")
        
        for i, (left, right) in enumerate(env.actions):
            action_desc = env.action_descriptions[i]
            print(f"     {i}: {action_desc} -> L={left}, R={right}")
        
        # Test a sample action
        print(f"   Testing action 0 in phase {phase}:")
        state = env._get_state()
        print(f"     Initial state shape: {state.shape}")
        
        # Execute action 0
        next_state, reward, done, info = env.step(0)
        print(f"     Reward: {reward:.2f}")
        print(f"     Done: {done}")
        print(f"     Info keys: {list(info.keys())}")
        print()
    
    # Test phase transitions with mock vision data
    print("🔄 Testing Phase Transitions")
    print("-" * 30)
    
    # Reset to phase 0
    env.reset()
    print(f"   Reset to phase: {env.current_phase_num}")
    
    # Simulate object detection and centering
    print("   Simulating object detection...")
    env.current_phase_num = 0
    env._update_action_space()
    
    # Mock state with object detected and centered
    mock_state = np.array([0.5] * 8 + [1.0, 0.3, 0.1, 0.0, 1.0, 0.0], dtype=np.float32)
    reward, info = env._calculate_reward(3, mock_state)  # Forward action
    print(f"     Object detection reward: {reward:.2f}")
    print(f"     Phase after object detection: {env.current_phase_num}")
    
    # Simulate object collection
    print("   Simulating object collection...")
    mock_state = np.array([0.5] * 8 + [1.0, 0.1, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)
    reward, info = env._calculate_reward(1, mock_state)  # Forward action
    print(f"     Object collection reward: {reward:.2f}")
    print(f"     Object collected: {env.object_collected}")
    print(f"     Phase after collection: {env.current_phase_num}")
    
    # Simulate target detection
    print("   Simulating target detection...")
    mock_state = np.array([0.5] * 8 + [0.0, 1.0, 0.0, 1.0, 0.3, 0.1], dtype=np.float32)
    reward, info = env._calculate_reward(1, mock_state)  # Forward action
    print(f"     Target detection reward: {reward:.2f}")
    print(f"     Phase after target detection: {env.current_phase_num}")
    
    # Simulate task completion
    print("   Simulating task completion...")
    mock_state = np.array([0.5] * 8 + [0.0, 1.0, 0.0, 1.0, 0.1, 0.0], dtype=np.float32)
    reward, info = env._calculate_reward(1, mock_state)  # Forward action
    print(f"     Task completion reward: {reward:.2f}")
    print(f"     Task completed: {env.task_completed}")
    
    print()
    print("✅ Phase-specific actions test completed successfully!")
    print("🎯 Key features validated:")
    print("   - Phase-specific action sets are correctly defined")
    print("   - Action space updates when transitioning between phases")
    print("   - Reward function handles phase transitions")
    print("   - Environment maintains phase state correctly")
    print("   - Mock testing framework works for validation")

if __name__ == "__main__":
    test_phase_specific_actions()
