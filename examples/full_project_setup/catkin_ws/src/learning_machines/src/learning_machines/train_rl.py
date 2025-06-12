#!/usr/bin/env python3
"""
Unified RL Training Script for Task 1 - Obstacle Avoidance

This script trains any RL agent (Q-Learning, DQN, Policy Gradient, Actor-Critic) 
for obstacle avoidance using a unified interface.

Usage:
    python train_rl.py --method policy_gradient --episodes 300 --learning-rate 0.0005
    python train_rl.py --method dqn --episodes 500 --batch-size 64
    python train_rl.py --method qlearning --episodes 1000 --epsilon-decay 0.995
    python train_rl.py --method actor_critic --episodes 400 --value-loss-coef 0.5
"""

import argparse
import sys
import os
from pathlib import Path

# Try to import rospy, but don't fail if not available
try:
    import rospy
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False

# Add the learning_machines package to the path
sys.path.append('/root/catkin_ws/src/learning_machines/src')

from learning_machines.test_actions import (
    SimulationRobobo, 
    HardwareRobobo,
    rl_obstacle_avoidance_task1
)
from data_files import FIGURES_DIR

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Train RL agent for obstacle avoidance',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --method policy_gradient --episodes 300
  %(prog)s --method dqn --episodes 500 --learning-rate 0.001
  %(prog)s --method qlearning --episodes 1000 --epsilon-decay 0.995
  %(prog)s --method actor_critic --episodes 400 --gamma 0.99
        """
    )
    
    # Required arguments
    parser.add_argument('--method', type=str, required=True,
                       choices=['qlearning', 'dqn', 'policy_gradient', 'actor_critic'],
                       help='RL method to train')
    
    # Platform selection (required)
    platform_group = parser.add_mutually_exclusive_group(required=True)
    platform_group.add_argument('--simulation', action='store_true',
                               help='Use simulation robot (CoppeliaSim)')
    platform_group.add_argument('--hardware', action='store_true',
                               help='Use hardware robot')
    
    # Training parameters
    parser.add_argument('--episodes', type=int, default=200, 
                       help='Number of training episodes (default: 200)')
    parser.add_argument('--max-steps', type=int, default=500,
                       help='Maximum steps per episode (default: 500)')
    parser.add_argument('--mode', type=str, default='train',
                       choices=['train', 'evaluate', 'train_and_evaluate'],
                       help='Training mode (default: train)')
    
    # Common RL parameters
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Learning rate (default: 0.001)')
    parser.add_argument('--gamma', type=float, default=0.95,
                       help='Discount factor (default: 0.95)')
    parser.add_argument('--epsilon', type=float, default=1.0,
                       help='Initial epsilon for exploration (default: 1.0)')
    parser.add_argument('--epsilon-decay', type=float, default=0.995,
                       help='Epsilon decay rate (default: 0.995)')
    parser.add_argument('--epsilon-min', type=float, default=0.01,
                       help='Minimum epsilon (default: 0.01)')
    
    # DQN-specific parameters
    parser.add_argument('--memory-size', type=int, default=10000,
                       help='Experience replay buffer size (default: 10000)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Training batch size for DQN (default: 32)')
    parser.add_argument('--target-update-freq', type=int, default=100,
                       help='Target network update frequency (default: 100)')
    
    # Actor-Critic specific parameters
    parser.add_argument('--value-loss-coef', type=float, default=0.5,
                       help='Value function loss coefficient (default: 0.5)')
    parser.add_argument('--entropy-coef', type=float, default=0.01,
                       help='Entropy bonus coefficient (default: 0.01)')
    
    # Model management
    parser.add_argument('--save-freq', type=int, default=50,
                       help='Save model every N episodes (default: 50)')
    parser.add_argument('--load-model', type=str, default=None,
                       help='Path to load pre-trained model')
    
    # Environment thresholds
    parser.add_argument('--obstacle-threshold', type=float, default=None,
                       help='Obstacle detection threshold (simulation: 150, hardware: 20)')
    
    return parser.parse_args()

def validate_args(args):
    """Validate argument combinations"""
    if args.method == 'dqn' and args.batch_size > args.memory_size:
        print(f"Warning: batch_size ({args.batch_size}) > memory_size ({args.memory_size})")
        print("Reducing batch_size to memory_size/4")
        args.batch_size = args.memory_size // 4
    
    if args.learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    
    if not 0 <= args.gamma <= 1:
        raise ValueError("gamma must be between 0 and 1")
    
    if not 0 <= args.epsilon <= 1:
        raise ValueError("epsilon must be between 0 and 1")

def print_config(args):
    """Print training configuration"""
    print("=" * 80)
    print(f"RL TRAINING - TASK 1 OBSTACLE AVOIDANCE")
    print("=" * 80)
    print(f"Method: {args.method.upper()}")
    print(f"Platform: {'Simulation (CoppeliaSim)' if args.simulation else 'Hardware Robot'}")
    print(f"Mode: {args.mode}")
    print(f"Episodes: {args.episodes}")
    print(f"Max steps per episode: {args.max_steps}")
    print(f"Results directory: {FIGURES_DIR}")
    print()
    
    print("Hyperparameters:")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Discount factor (gamma): {args.gamma}")
    
    if args.method in ['qlearning', 'dqn']:
        print(f"  Initial epsilon: {args.epsilon}")
        print(f"  Epsilon decay: {args.epsilon_decay}")
        print(f"  Minimum epsilon: {args.epsilon_min}")
    
    if args.method == 'dqn':
        print(f"  Memory size: {args.memory_size}")
        print(f"  Batch size: {args.batch_size}")
        print(f"  Target update frequency: {args.target_update_freq}")
    
    if args.method == 'actor_critic':
        print(f"  Value loss coefficient: {args.value_loss_coef}")
        print(f"  Entropy coefficient: {args.entropy_coef}")
    
    print(f"  Save frequency: {args.save_freq} episodes")
    if args.load_model:
        print(f"  Load model: {args.load_model}")
    print()

def main():
    """Main training function"""
    args = parse_arguments()
    validate_args(args)
    print_config(args)
    
    # Initialize ROS node only for hardware mode
    if args.hardware and ROS_AVAILABLE:
        rospy.init_node(f'{args.method}_training', anonymous=True)
        use_ros = True
    else:
        use_ros = False
        if args.hardware:
            print("Warning: ROS not available, running in simulation mode")
    
    try:
        # Create robot interface based on platform
        if args.simulation:
            print("Connecting to CoppeliaSim simulation...")
            rob = SimulationRobobo()
            # Wait for connection (simulation doesn't need ROS)
            import time
            time.sleep(2)
        else:
            print("Connecting to hardware robot...")
            rob = HardwareRobobo()
            # Wait for connection
            if use_ros and ROS_AVAILABLE:
                rospy.sleep(2)
            else:
                import time
                time.sleep(2)
        
        print("Connected successfully!")
        
        # Start training
        print(f"\nStarting {args.method.upper()} training...")
        
        # Prepare training arguments
        rl_kwargs = {
            'rob': rob,
            'agent_type': args.method,
            'mode': args.mode,
            'num_episodes': args.episodes,
            'max_steps_per_episode': args.max_steps
        }
        
        # Add threshold arguments if provided
        if args.obstacle_threshold is not None:
            rl_kwargs['obstacle_threshold'] = args.obstacle_threshold
        
        results = rl_obstacle_avoidance_task1(**rl_kwargs)
        
        print("\n" + "=" * 80)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        
        if results:
            if args.mode == 'train':
                agent, metrics = results
                print_training_summary(metrics, args.method)
            elif args.mode == 'evaluate':
                eval_metrics = results
                print_evaluation_summary(eval_metrics, args.method)
            elif args.mode == 'train_and_evaluate':
                agent, training_metrics, eval_metrics = results
                print_training_summary(training_metrics, args.method)
                print_evaluation_summary(eval_metrics, args.method)
            
            print(f"\nResults saved to: {FIGURES_DIR}")
            print_saved_files(args.method)
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        import traceback
        traceback.print_exc()
        
    finally:
        print("\nShutting down...")
        if use_ros and ROS_AVAILABLE:
            rospy.signal_shutdown("Training completed")

def print_training_summary(metrics, method):
    """Print training statistics summary"""
    print(f"\nTraining Summary ({method.upper()}):")
    print(f"  Total episodes: {len(metrics['episode_rewards'])}")
    
    if metrics['episode_rewards']:
        last_20_avg = sum(metrics['episode_rewards'][-20:]) / min(20, len(metrics['episode_rewards']))
        print(f"  Average reward (last 20 episodes): {last_20_avg:.2f}")
        print(f"  Best episode reward: {max(metrics['episode_rewards']):.2f}")
        print(f"  Worst episode reward: {min(metrics['episode_rewards']):.2f}")
    
    if metrics['episode_lengths']:
        avg_length = sum(metrics['episode_lengths']) / len(metrics['episode_lengths'])
        print(f"  Average episode length: {avg_length:.1f}")
    
    if metrics['collision_rates']:
        avg_collision_rate = sum(metrics['collision_rates']) / len(metrics['collision_rates'])
        print(f"  Average collision rate: {avg_collision_rate:.3f}")

def print_evaluation_summary(eval_metrics, method):
    """Print evaluation statistics summary"""
    print(f"\nEvaluation Summary ({method.upper()}):")
    print(f"  Average reward: {eval_metrics['average_reward']:.2f}")
    print(f"  Success rate (no collisions): {eval_metrics['success_rate']:.2%}")
    print(f"  Average episode length: {eval_metrics['average_length']:.2f}")
    print(f"  Average collisions per episode: {sum(eval_metrics['collision_counts']) / len(eval_metrics['collision_counts']):.2f}")

def print_saved_files(method):
    """Print information about saved files"""
    print("Saved files:")
    print(f"  - Model weights: rl_model_{method}_*.pth")
    print(f"  - Training metrics: rl_metrics_{method}_*.json")
    print(f"  - Training plots: rl_training_{method}_*.png")

if __name__ == '__main__':
    main()
