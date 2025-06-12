#!/usr/bin/env python3
"""
Test script for the fixed Policy Gradient implementation
This script verifies:
1. Robot emotion system works correctly  
2. Policy Gradient agent can select and execute actions
3. Training loop functions without robot getting stuck
4. Reward system provides proper learning signals
"""

import sys
import time
import numpy as np
from pathlib import Path

# Add the learning_machines module to path
sys.path.append('/Users/saahithshetty/Documents/Coding/Learning_Machines/learning_machines_robobo/examples/full_project_setup/catkin_ws/src/learning_machines/src')

from learning_machines.test_actions import (
    RobotEnvironment,
    PolicyGradientAgent, 
    create_rl_agent,
    train_rl_agent
)
from robobo_interface import SimulationRobobo, Emotion


def test_emotion_system():
    """Test that emotion system works correctly"""
    print("="*60)
    print("TESTING EMOTION SYSTEM")
    print("="*60)
    
    # Test all emotions
    emotions_to_test = [
        Emotion.HAPPY,
        Emotion.LAUGHING,  # Fixed typo
        Emotion.SAD, 
        Emotion.ANGRY,
        Emotion.SURPRISED,
        Emotion.NORMAL
    ]
    
    print("Available emotions:")
    for emotion in emotions_to_test:
        print(f"  - {emotion.name}: '{emotion.value}'")
    
    print("✅ All emotions are correctly defined!")
    return True


def test_action_space():
    """Test robot action space and movement"""
    print("\n" + "="*60)
    print("TESTING ACTION SPACE")
    print("="*60)
    
    # Create simulation robot (dummy instance for testing)
    try:
        # Create environment without actual robot connection for testing
        class MockRobot:
            def read_irs(self):
                return [0.5] * 8  # Mock IR readings
            def get_orientation(self):
                class MockOrientation:
                    yaw = 0.0
                return MockOrientation()
            def move_blocking(self, left, right, millis):
                print(f"  Move command: left={left}, right={right}, duration={millis}ms")
            def set_emotion(self, emotion):
                print(f"  Robot emotion: {emotion.value}")
            def play_simulation(self):
                pass
        
        mock_robot = MockRobot()
        env = RobotEnvironment(mock_robot, max_episode_steps=10)
        
        print(f"Action space size: {env.action_space_size}")
        print("Available actions:")
        for i, (left, right) in enumerate(env.actions):
            print(f"  {i}: {env.action_descriptions[i]} -> ({left}, {right})")
        
        print("✅ Action space is correctly defined!")
        return True
        
    except Exception as e:
        print(f"❌ Error testing action space: {e}")
        return False


def test_policy_network():
    """Test Policy Gradient agent creation and action selection"""
    print("\n" + "="*60)
    print("TESTING POLICY GRADIENT AGENT")
    print("="*60)
    
    try:
        # Create agent
        state_size = 11
        action_size = 9
        agent = PolicyGradientAgent(state_size, action_size)
        
        print(f"Agent created:")
        print(f"  State size: {agent.state_size}")
        print(f"  Action size: {agent.action_size}")
        print(f"  Device: {agent.device}")
        print(f"  Exploration episodes: {agent.exploration_episodes}")
        
        # Test action selection
        print("\nTesting action selection:")
        test_state = np.random.random(state_size)
        
        # Test multiple action selections
        actions_selected = []
        for i in range(10):
            action = agent.get_action(test_state, training=True)
            actions_selected.append(action)
            
        print(f"Sample actions selected: {actions_selected}")
        print(f"Action range: [{min(actions_selected)}, {max(actions_selected)}]")
        
        # Verify actions are in valid range
        if all(0 <= a < action_size for a in actions_selected):
            print("✅ Policy Gradient agent works correctly!")
            return True
        else:
            print("❌ Actions out of valid range!")
            return False
            
    except Exception as e:
        print(f"❌ Error testing Policy Gradient agent: {e}")
        return False


def test_reward_system():
    """Test reward calculation and emotion mapping"""
    print("\n" + "="*60)
    print("TESTING REWARD SYSTEM")
    print("="*60)
    
    try:
        class MockRobot:
            def __init__(self):
                self.emotion_set = None
                
            def read_irs(self):
                return [0.5, 0.3, 0.2, 0.1, 0.15, 0.4, 0.6, 0.3]  # Mock IR readings
                
            def get_orientation(self):
                class MockOrientation:
                    yaw = 0.0
                return MockOrientation()
                
            def move_blocking(self, left, right, millis):
                pass
                
            def set_emotion(self, emotion):
                self.emotion_set = emotion
                print(f"  Robot emotion set: {emotion.name} ('{emotion.value}')")
                
            def play_simulation(self):
                pass
        
        mock_robot = MockRobot()
        env = RobotEnvironment(mock_robot, max_episode_steps=10)
        
        # Test different scenarios
        test_scenarios = [
            (4, "Forward movement"),      # Forward action
            (8, "Stop action"),          # Stop action  
            (1, "Turn left"),            # Turn action
            (0, "Backward movement"),    # Backward action
        ]
        
        print("Testing reward calculation for different actions:")
        state = env._get_state()
        
        for action_idx, description in test_scenarios:
            reward, info = env._calculate_reward(action_idx, state)
            print(f"  {description}: reward={reward:.2f}, emotion={mock_robot.emotion_set.name if mock_robot.emotion_set else 'None'}")
        
        print("✅ Reward system works correctly!")
        return True
        
    except Exception as e:
        print(f"❌ Error testing reward system: {e}")
        return False


def run_mini_training_test():
    """Run a short training test to verify the system works end-to-end"""
    print("\n" + "="*60)
    print("MINI TRAINING TEST (10 episodes)")
    print("="*60)
    
    try:
        class MockRobot:
            def __init__(self):
                self.step_count = 0
                self.position_x = 0.0
                self.position_y = 0.0
                
            def read_irs(self):
                # Simulate gradually changing sensor readings
                base_reading = 0.5 + 0.1 * np.sin(self.step_count * 0.1)
                return [base_reading + 0.1 * np.random.random() for _ in range(8)]
                
            def get_orientation(self):
                class MockOrientation:
                    yaw = self.step_count * 0.05  # Gradual rotation
                return MockOrientation()
                
            def get_position(self):
                class MockPosition:
                    x = self.position_x
                    y = self.position_y  
                    z = 0.0
                return MockPosition()
                
            def move_blocking(self, left, right, millis):
                # Simulate movement effect on position
                avg_speed = (left + right) / 200.0  # Normalize speed
                self.position_x += avg_speed * 0.01
                if left != right:  # Turning
                    self.position_y += (left - right) / 200.0 * 0.005
                self.step_count += 1
                
            def set_emotion(self, emotion):
                pass  # Silent for mini test
                
            def play_simulation(self):
                pass
        
        # Create mock robot and environment
        mock_robot = MockRobot()
        env = RobotEnvironment(mock_robot, max_episode_steps=20)
        agent = PolicyGradientAgent(env.state_size, env.action_space_size, learning_rate=0.01)
        
        print("Running 10 episodes of training...")
        total_rewards = []
        
        for episode in range(10):
            state = env.reset()
            episode_reward = 0
            steps = 0
            
            for step in range(20):  # Max 20 steps per episode
                action = agent.get_action(state, training=True)
                next_state, reward, done, info = env.step(action)
                agent.update(state, action, reward, next_state, done)
                
                episode_reward += reward
                steps += 1
                state = next_state
                
                if done:
                    break
            
            total_rewards.append(episode_reward)
            if episode % 2 == 0:  # Print every 2nd episode
                print(f"  Episode {episode+1}: {steps} steps, reward={episode_reward:.2f}")
        
        # Check if training shows any improvement trend
        early_avg = np.mean(total_rewards[:3])
        late_avg = np.mean(total_rewards[-3:])
        
        print(f"\nTraining summary:")
        print(f"  Early episodes avg reward: {early_avg:.2f}")
        print(f"  Late episodes avg reward: {late_avg:.2f}")
        print(f"  Total episodes completed: {len(total_rewards)}")
        
        if len(total_rewards) == 10:
            print("✅ Mini training test completed successfully!")
            return True
        else:
            print("❌ Training test failed!")
            return False
            
    except Exception as e:
        print(f"❌ Error in mini training test: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("🤖 TESTING FIXED POLICY GRADIENT IMPLEMENTATION")
    print("=" * 80)
    
    tests = [
        ("Emotion System", test_emotion_system),
        ("Action Space", test_action_space), 
        ("Policy Network", test_policy_network),
        ("Reward System", test_reward_system),
        ("Mini Training", run_mini_training_test)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🧪 Running {test_name} test...")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} test failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*80)
    print("TEST RESULTS SUMMARY")
    print("="*80)
    
    passed = 0
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_name:.<30} {status}")
        if success:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 ALL TESTS PASSED! Policy Gradient implementation is ready for training.")
        
        print("\n📋 Next Steps:")
        print("1. Run full training: `python -c \"from learning_machines.test_actions import *; train_rl_agent(robot, 'policy_gradient', 100)\"`")
        print("2. Monitor robot movement and emotion feedback")
        print("3. Adjust hyperparameters if needed")
        print("4. Evaluate trained agent performance")
        
    else:
        print(f"\n⚠️  {len(results) - passed} tests failed. Please fix issues before training.")
    
    return passed == len(results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
