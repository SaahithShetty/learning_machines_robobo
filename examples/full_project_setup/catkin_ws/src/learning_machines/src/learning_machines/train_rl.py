#!/usr/bin/env python3
"""
Unified RL Training Script for Task 2 - Green Food Collection

This script trains any RL agent (Q-Learning, DQN, Policy Gradient, Actor-Critic) 
for green food collection using computer vision and reinforcement learning.

Usage:
    python train_rl.py --simulation --method dqn --episodes 100
    python train_rl.py --simulation --method qlearning --episodes 200
    python train_rl.py --simulation --method policy_gradient --episodes 150
    python train_rl.py --simulation --method actor_critic --episodes 100
    python train_rl.py --simulation --method dqn --mode evaluate --load-model model.pth --episodes 10
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
    green_food_collection_task2,
    test_task2_capabilities
)
from learning_machines.agent_factory import get_default_hyperparameters
from data_files import FIGURES_DIR

def parse_arguments():
    """Parse command line arguments"""
    
    # First pass: get method to determine defaults
    temp_parser = argparse.ArgumentParser(add_help=False)
    temp_parser.add_argument('--method', type=str, choices=['qlearning', 'dqn', 'policy_gradient', 'actor_critic'])
    temp_args, _ = temp_parser.parse_known_args()
    
    # Get defaults from agent_factory based on method
    defaults = {}
    if temp_args.method:
        defaults = get_default_hyperparameters(temp_args.method)
    
    # Main parser with dynamic defaults
    parser = argparse.ArgumentParser(
        description='Train RL agent for green food collection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --simulation --method dqn --episodes 100
  %(prog)s --simulation --method qlearning --episodes 200
  %(prog)s --simulation --method policy_gradient --episodes 150
  %(prog)s --simulation --method actor_critic --episodes 100
  %(prog)s --simulation --method dqn --mode evaluate --load-model model.pth --episodes 10
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
    parser.add_argument('--episodes', type=int, default=100, 
                       help='Number of training episodes (default: 100)')
    parser.add_argument('--mode', type=str, default='train',
                       choices=['train', 'evaluate', 'train_and_evaluate', 'test_capabilities'],
                       help='Training mode (default: train)')
    
    # Model loading/saving
    parser.add_argument('--load-model', type=str, default=None,
                       help='Path to pre-trained model file (for evaluation or continued training)')
    parser.add_argument('--save-model', type=str, default=None,
                       help='Path to save trained model (default: auto-generated)')
    
    # Common RL parameters - using agent_factory defaults
    parser.add_argument('--learning-rate', type=float, default=defaults.get('learning_rate', 0.001),
                       help=f'Learning rate (default: {defaults.get("learning_rate", 0.001)})')
    parser.add_argument('--gamma', type=float, default=defaults.get('gamma', 0.95),
                       help=f'Discount factor (default: {defaults.get("gamma", 0.95)})')
    parser.add_argument('--epsilon', type=float, default=defaults.get('epsilon', 1.0),
                       help=f'Initial epsilon for exploration (default: {defaults.get("epsilon", 1.0)})')
    parser.add_argument('--epsilon-decay', type=float, default=defaults.get('epsilon_decay', 0.995),
                       help=f'Epsilon decay rate (default: {defaults.get("epsilon_decay", 0.995)})')
    parser.add_argument('--epsilon-min', type=float, default=defaults.get('epsilon_min', 0.01),
                       help=f'Minimum epsilon for exploration (default: {defaults.get("epsilon_min", 0.01)})')
    
    # Hardware/Simulation specific parameters
    parser.add_argument('--collision-threshold', type=float, default=None,
                       help='IR sensor collision threshold (default: 0.05 for simulation, 20.0 for hardware)')
    
    # DQN-specific parameters - using agent_factory defaults
    parser.add_argument('--memory-size', type=int, default=defaults.get('memory_size', 10000),
                       help=f'Experience replay buffer size (default: {defaults.get("memory_size", 10000)})')
    parser.add_argument('--batch-size', type=int, default=defaults.get('batch_size', 32),
                       help=f'Training batch size for DQN (default: {defaults.get("batch_size", 32)})')
    parser.add_argument('--target-update-freq', type=int, default=defaults.get('target_update_freq', 100),
                       help=f'Target network update frequency (default: {defaults.get("target_update_freq", 100)})')
    
    # Actor-Critic specific parameters
    parser.add_argument('--value-loss-coef', type=float, default=0.5,
                       help='Value function loss coefficient (default: 0.5)')
    parser.add_argument('--entropy-coef', type=float, default=0.01,
                       help='Entropy bonus coefficient (default: 0.01)')
    
    # Model management
    parser.add_argument('--save-freq', type=int, default=50,
                       help='Save model every N episodes (default: 50)')
    
    # Optional: Legacy threshold support for testing (NOT recommended)
    parser.add_argument('--use-thresholds', action='store_true',
                       help='Use legacy threshold-based detection instead of unified proximity (NOT recommended)')
    
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
    print(f"RL TRAINING - TASK 2: GREEN FOOD COLLECTION")
    print("=" * 80)
    print(f"Method: {args.method.upper()}")
    print(f"Platform: {'Simulation (CoppeliaSim)' if args.simulation else 'Hardware Robot'}")
    print(f"Mode: {args.mode}")
    print(f"Episodes: {args.episodes}")
    print(f"Episode duration: 3 minutes (180 seconds) - Task 2 time limit")
    
    # Show collision threshold info
    if args.collision_threshold is not None:
        threshold_info = f"Custom: {args.collision_threshold}"
    else:
        threshold_info = f"Auto: {0.05 if args.simulation else 20.0} ({'simulation' if args.simulation else 'hardware'} default)"
    print(f"Collision threshold: {threshold_info}")
    
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
            print("Connecting to simulation robot...")
            rob = SimulationRobobo()
            # Wait for connection (simulation doesn't need ROS)
            import time
            time.sleep(2)
        else:
            print("Connecting to hardware robot...")
            rob = HardwareRobobo(camera=True)
            # Wait for connection
            if use_ros and ROS_AVAILABLE:
                rospy.sleep(2)
            else:
                import time
                time.sleep(2)
        
        print("Connected successfully!")
        
        # Handle test capabilities mode
        if args.mode == 'test_capabilities':
            print(f"\nRunning capability tests on {'simulation' if args.simulation else 'hardware'}...")
            test_task2_capabilities(rob)
            return
        
        # Start training
        print(f"\nStarting {args.method.upper()} training for green food collection...")
        
        # Prepare training arguments for Task 2
        rl_kwargs = {
            'rob': rob,
            'agent_type': args.method,
            'mode': args.mode,
            'num_episodes': args.episodes
        }
        
        # Set collision threshold based on platform
        if args.collision_threshold is not None:
            collision_threshold = args.collision_threshold
        else:
            # Auto-detect threshold based on platform
            collision_threshold = 0.05 if args.simulation else 20.0
        
        rl_kwargs['collision_threshold'] = collision_threshold
        print(f"Using collision threshold: {collision_threshold} ({'simulation' if args.simulation else 'hardware'} default)")
        
        # Add model path for evaluation mode
        if args.load_model is not None:
            # Convert host path to container path if needed
            model_path = args.load_model
            original_path = model_path
            
            if model_path.startswith('./results/'):
                # Convert relative host path to absolute container path
                model_path = model_path.replace('./results/', '/root/results/')
            elif '/results/figures/' in model_path and not model_path.startswith('/root/'):
                # Convert absolute host path to container path
                model_path = model_path.split('/results/figures/')[-1]
                model_path = f'/root/results/figures/{model_path}'
                
            print(f"Model path conversion: {original_path} -> {model_path}")
            rl_kwargs['model_path'] = model_path
        
        results = green_food_collection_task2(**rl_kwargs)
        
        print("\n" + "=" * 80)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        
        if results:
            if args.mode == 'train':
                print_training_summary(results, args.method)
            elif args.mode == 'evaluate':
                print_evaluation_summary(results, args.method)
            elif args.mode == 'train_and_evaluate':
                print_training_summary(results, args.method)
                # Note: eval metrics would be separate if implemented
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
    """Print training statistics summary for food collection"""
    print(f"\nTraining Summary ({method.upper()}) - Green Food Collection:")
    print(f"  Total episodes: {len(metrics['episode_rewards'])}")
    
    if metrics['episode_rewards']:
        last_20_avg = sum(metrics['episode_rewards'][-20:]) / min(20, len(metrics['episode_rewards']))
        print(f"  Average reward (last 20 episodes): {last_20_avg:.2f}")
        print(f"  Best episode reward: {max(metrics['episode_rewards']):.2f}")
        print(f"  Worst episode reward: {min(metrics['episode_rewards']):.2f}")
    
    # Task 2 specific metrics
    if 'episode_food_collected' in metrics:
        avg_food = sum(metrics['episode_food_collected']) / len(metrics['episode_food_collected'])
        print(f"  Average foods collected per episode: {avg_food:.1f}/7")
        
    if 'success_rate' in metrics:
        success_rate = metrics['success_rate'][-1] if metrics['success_rate'] else 0
        print(f"  Success rate (7 foods collected): {success_rate:.1%}")
    
    if 'episode_times' in metrics:
        avg_time = sum(metrics['episode_times']) / len(metrics['episode_times'])
        print(f"  Average completion time: {avg_time:.1f} seconds")

def print_evaluation_summary(results, method):
    """Print evaluation statistics summary for food collection"""
    print(f"\nEvaluation Summary ({method.upper()}) - Green Food Collection:")
    
    # Handle the results dictionary structure from green_food_collection_task2
    if 'episode_rewards' in results and results['episode_rewards']:
        average_reward = sum(results['episode_rewards']) / len(results['episode_rewards'])
        print(f"  Average reward: {average_reward:.2f}")
    
    if 'episode_success_rates' in results and results['episode_success_rates']:
        success_rate = sum(results['episode_success_rates']) / len(results['episode_success_rates'])
        print(f"  Success rate (7 foods collected): {success_rate:.1%}")
    
    if 'episode_foods_collected' in results and results['episode_foods_collected']:
        avg_foods = sum(results['episode_foods_collected']) / len(results['episode_foods_collected'])
        print(f"  Average foods collected: {avg_foods:.1f}/7")
    
    if 'episode_lengths' in results and results['episode_lengths']:
        avg_time = sum(results['episode_lengths']) / len(results['episode_lengths'])
        print(f"  Average episode length: {avg_time:.1f} steps")

def print_saved_files(method):
    """Print information about saved files"""
    print("Saved files:")
    print(f"  - Model weights: rl_model_{method}_*.pth")
    print(f"  - Training metrics: rl_metrics_{method}_*.json")
    print(f"  - Training plots: rl_training_{method}_*.png")

if __name__ == '__main__':
    main()
