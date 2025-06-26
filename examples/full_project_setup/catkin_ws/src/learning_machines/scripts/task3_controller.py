#!/usr/bin/env python3
"""
Task 3: Object Pushing Controller
==================================

Controller script for running the object pushing to target location task using reinforcement learning
and computer vision.

This script provides a command-line interface for:
- Training RL agents for object pushing
- Evaluating trained models  
- Running demo sessions
- Testing vision systems

Usage Examples:
    # Train DQN agent for object pushing
    python3 task3_controller.py --simulation --method dqn --episodes 100

    # Evaluate trained model
    python3 task3_controller.py --simulation --method dqn --mode evaluate --load-model model.pth

    # Quick demo of all RL methods
    python3 task3_controller.py --simulation --demo-all

    # Test vision system
    python3 task3_controller.py --simulation --test-vision
"""

import sys
import argparse
from pathlib import Path

# Add the learning_machines package to the path
sys.path.append('/root/catkin_ws/src/learning_machines/src')

from learning_machines.test_actions import SimulationRobobo, HardwareRobobo
from learning_machines import (
    run_all_actions,
    demo_task3_object_pushing,
    test_task3_capabilities,
    test_object_vision_system
)

def main():
    parser = argparse.ArgumentParser(description='Task 3: Object Pushing Controller')
    
    # Platform selection
    platform_group = parser.add_mutually_exclusive_group(required=True)
    platform_group.add_argument('--hardware', action='store_true', help='Run on hardware robot')
    platform_group.add_argument('--simulation', action='store_true', help='Run in CoppeliaSim simulator')
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument('--demo-all', action='store_true', 
                           help='Demo all 4 RL methods with short training')
    mode_group.add_argument('--test-vision', action='store_true',
                           help='Test the computer vision system for object/target detection')
    mode_group.add_argument('--test-capabilities', action='store_true',
                           help='Test all Task 3 system capabilities')
    
    # RL Training options (default mode)
    parser.add_argument('--method', choices=['dqn', 'qlearning', 'policy_gradient', 'actor_critic'],
                        default='dqn', help='RL agent type (default: dqn)')
    parser.add_argument('--mode', choices=['train', 'evaluate', 'train_and_evaluate'],
                        default='train', help='Training mode (default: train)')
    parser.add_argument('--episodes', type=int, default=100, 
                        help='Number of episodes to run (default: 100)')
    parser.add_argument('--load-model', type=str,
                        help='Path to saved model for evaluation')
    
    # Additional options
    parser.add_argument('--collision-threshold', type=float, default=0.95,
                        help='IR sensor collision threshold (default: 0.95)')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output')
    parser.add_argument('--save-results', action='store_true', default=True,
                        help='Save training results and models (default: True)')
    
    args = parser.parse_args()
    
    # Create robot interface
    try:
        if args.hardware:
            print("🤖 Initializing hardware robot interface...")
            rob = HardwareRobobo(camera=True)
            platform = "hardware"
        else:
            print("🎮 Initializing simulation robot interface...")
            rob = SimulationRobobo()
            platform = "simulation"
            
        print(f"✅ Robot interface ready on {platform}")
        
        # Execute requested mode
        if args.demo_all:
            print("\n🎯 Running demo of all 4 RL methods...")
            demo_task3_object_pushing(rob)
            
        elif args.test_vision:
            print("\n👁️  Testing computer vision system...")
            test_object_vision_system(rob)
            
        elif args.test_capabilities:
            print("\n🔧 Testing all Task 3 capabilities...")
            test_task3_capabilities(rob)
            
        else:
            # Default: RL training/evaluation
            print(f"\n🧠 Running RL object pushing with {args.method.upper()}")
            print(f"Mode: {args.mode}, Episodes: {args.episodes}")
            
            results = run_all_actions(
                rob=rob,
                rl_agent_type=args.method,
                rl_mode=args.mode,
                rl_episodes=args.episodes,
                collision_threshold=args.collision_threshold
            )
            
            print("\n📊 Final Results:")
            if 'success_rate' in results:
                print(f"  Success Rate: {results['success_rate']:.1%}")
            if 'episode_completion_count' in results and results['episode_completion_count']:
                import numpy as np
                completion_rate = np.mean(results['episode_completion_count'])
                max_completion = max(results['episode_completion_count'])
                print(f"  Average Task Completion: {completion_rate:.1%}")
                print(f"  Best Episode: {max_completion} task completions")
            
            if args.save_results and args.mode in ['train', 'train_and_evaluate']:
                print("💾 Results saved to results/figures/ directory")
                
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
        return 1
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1
        
    finally:
        # Cleanup
        try:
            if 'rob' in locals():
                rob.move_blocking(0, 0, 100)  # Stop robot
                if hasattr(rob, 'stop_simulation'):
                    rob.stop_simulation()
            print("\n🧹 Cleanup completed")
        except:
            pass
            
    print("\n✅ Task 3 controller completed successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
