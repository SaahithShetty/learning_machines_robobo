#!/usr/bin/env python3
"""
Policy Gradient (REINFORCE) Training Script for Task 1 - Obstacle Avoidance

This script trains a Policy Gradient agent to perform obstacle avoidance using the REINFORCE algorithm.
The agent learns to navigate through an environment with obstacles while maximizing distance traveled
and minimizing collisions.

Usage:
    python train_policy_gradient.py [--episodes 200] [--max-steps 500] [--learning-rate 0.001]
"""

import rospy
import argparse
import sys
import os
from pathlib import Path

# Add the learning_machines package to the path
sys.path.append('/root/catkin_ws/src/learning_machines/src')

from learning_machines.test_actions import (
    SimulationRobobo, 
    rl_obstacle_avoidance_task1,
    FIGURES_DIR
)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train Policy Gradient agent for obstacle avoidance')
    parser.add_argument('--episodes', type=int, default=200, 
                       help='Number of training episodes (default: 200)')
    parser.add_argument('--max-steps', type=int, default=500,
                       help='Maximum steps per episode (default: 500)')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Learning rate for policy network (default: 0.001)')
    parser.add_argument('--gamma', type=float, default=0.95,
                       help='Discount factor (default: 0.95)')
    parser.add_argument('--save-freq', type=int, default=50,
                       help='Save model every N episodes (default: 50)')
    return parser.parse_args()

def main():
    """Main training function"""
    args = parse_arguments()
    
    print("=" * 80)
    print("POLICY GRADIENT (REINFORCE) TRAINING - TASK 1 OBSTACLE AVOIDANCE")
    print("=" * 80)
    print(f"Training Configuration:")
    print(f"  Episodes: {args.episodes}")
    print(f"  Max steps per episode: {args.max_steps}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Discount factor (gamma): {args.gamma}")
    print(f"  Save frequency: {args.save_freq} episodes")
    print(f"  Results directory: {FIGURES_DIR}")
    print()
    
    # Initialize ROS node
    rospy.init_node('policy_gradient_training', anonymous=True)
    
    try:
        # Create robot interface
        print("Connecting to CoppeliaSim...")
        rob = SimulationRobobo()
        
        # Wait for connection
        rospy.sleep(2)
        print("Connected successfully!")
        
        # Start training
        print("\nStarting Policy Gradient training...")
        results = rl_obstacle_avoidance_task1(
            rob=rob,
            agent_type='policy_gradient',
            mode='train',
            num_episodes=args.episodes
        )
        
        print("\n" + "=" * 80)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        
        if results:
            agent, metrics = results
            
            # Print final training statistics
            print(f"\nFinal Training Statistics:")
            print(f"  Total episodes: {len(metrics['episode_rewards'])}")
            print(f"  Average reward (last 20 episodes): {sum(metrics['episode_rewards'][-20:]) / min(20, len(metrics['episode_rewards'])):.2f}")
            print(f"  Best episode reward: {max(metrics['episode_rewards']):.2f}")
            print(f"  Average episode length: {sum(metrics['episode_lengths']) / len(metrics['episode_lengths']):.1f}")
            print(f"  Average collision rate: {sum(metrics['collision_rates']) / len(metrics['collision_rates']):.3f}")
            
            print(f"\nResults saved to: {FIGURES_DIR}")
            print("  - Model weights: rl_model_policy_gradient_*.pth")
            print("  - Training metrics: rl_metrics_policy_gradient_*.json")
            print("  - Training plots: rl_training_policy_gradient_*.png")
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        import traceback
        traceback.print_exc()
        
    finally:
        print("\nShutting down...")
        rospy.signal_shutdown("Training completed")
    finally:
        # Clean shutdown
        if hasattr(rob, 'stop_simulation'):
            rob.stop_simulation()
        print("Training session ended")

if __name__ == '__main__':
    main()
