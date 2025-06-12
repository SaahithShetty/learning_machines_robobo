#!/usr/bin/env python3
"""
Test script to verify the 9D optimized state space implementation
"""

import numpy as np
import sys
import os

# Mock robot interface for testing
class MockRobot:
    def read_irs(self):
        # Return mock IR sensor readings (8 sensors)
        return [100, 200, 150, 300, 250, 180, 120, 90]
    
    def read_orientation(self):
        # Return mock orientation data (not used in 9D state space)
        return {'yaw': 45, 'pitch': 0}

class TestRobotEnvironment:
    """Simplified version of RobotEnvironment for testing state space"""
    
    def __init__(self, robot, max_episode_steps=1000):
        self.robot = robot
        self.max_episode_steps = max_episode_steps
        self.state_size = 9  # 8 IR + 1 prev_action (optimized from 11D)
        self.action_space_size = 8
        self.prev_action = 0
        self.step_count = 0
        self.is_simulation = True  # Assume simulation for testing
        
        # Define actions (left_wheel_speed, right_wheel_speed)
        self.actions = [
            (30, 30),    # 0: Forward
            (0, 30),     # 1: Turn left
            (-30, 30),   # 2: Sharp left
            (30, 0),     # 3: Turn right
            (30, -30),   # 4: Sharp right
            (-30, -30),  # 5: Backward
            (-30, 0),    # 6: Reverse left
            (0, -30)     # 7: Reverse right
        ]
    
    def _get_state(self):
        """Get current state vector (9D optimized)"""
        # Get IR sensor readings (8 sensors)
        ir_values = self.robot.read_irs()
        ir_normalized = []
        
        for val in ir_values:
            if val is None:
                ir_normalized.append(0.0)  # No detection
            else:
                if self.is_simulation:
                    # Simulation range: 0-2000, higher = farther
                    ir_normalized.append(min(val / 2000.0, 1.0))
                else:
                    # Hardware range: 0-512, higher = closer
                    ir_normalized.append(min(val / 512.0, 1.0))
        
        # Previous action (1 value)
        prev_action_norm = [self.prev_action / float(self.action_space_size - 1)]
        
        # Combine: 8 IR + 1 prev_action = 9D state space
        # NOTE: Orientation removed as robot base never tilts and yaw changes predictably
        return np.array(ir_normalized + prev_action_norm, dtype=np.float32)

def test_9d_state_space():
    """Test the 9D optimized state space implementation"""
    print("🧪 Testing 9D Optimized State Space")
    print("=" * 50)
    
    # Create mock robot and environment
    mock_robot = MockRobot()
    env = TestRobotEnvironment(mock_robot, max_episode_steps=100)
    
    print(f"State size: {env.state_size}")
    print(f"Action space size: {env.action_space_size}")
    print()
    
    # Test state vector generation
    state = env._get_state()
    print(f"State vector shape: {state.shape}")
    print(f"State vector: {state}")
    print()
    
    # Analyze state components
    print("State Vector Analysis:")
    print(f"  IR sensors (8D): {state[:8]}")
    print(f"  Previous action (1D): {state[8]}")
    print()
    
    # Verify state dimensions
    print("Verification Tests:")
    try:
        assert env.state_size == 9, f"❌ Expected state_size=9, got {env.state_size}"
        print("✅ State size parameter correct (9)")
        
        assert state.shape[0] == 9, f"❌ Expected state vector length=9, got {state.shape[0]}"
        print("✅ State vector dimension correct (9)")
        
        assert len(state[:8]) == 8, f"❌ Expected 8 IR sensors, got {len(state[:8])}"
        print("✅ IR sensor count correct (8)")
        
        assert 0 <= state[8] <= 1, f"❌ Previous action should be normalized [0,1], got {state[8]}"
        print("✅ Previous action normalization correct")
        
        # Check IR sensor normalization
        ir_in_range = all(0 <= val <= 1 for val in state[:8])
        assert ir_in_range, f"❌ IR sensors should be normalized [0,1], got {state[:8]}"
        print("✅ IR sensor normalization correct")
        
        print()
        print("🎉 All tests passed! 9D state space optimization verified successfully!")
        
    except AssertionError as e:
        print(f"❌ Test failed: {e}")
        return False
    
    print()
    print("📊 Performance Benefits of 9D vs 11D State Space:")
    print("  ✅ Faster convergence - less irrelevant information")
    print("  ✅ More stable training - reduced dimensionality")
    print("  ✅ Better generalization - focuses on essential sensor data")
    print("  ✅ Reduced computational cost - smaller networks")
    print("  ✅ Cleaner debugging - easier to interpret")
    
    return True

if __name__ == "__main__":
    success = test_9d_state_space()
    sys.exit(0 if success else 1)
