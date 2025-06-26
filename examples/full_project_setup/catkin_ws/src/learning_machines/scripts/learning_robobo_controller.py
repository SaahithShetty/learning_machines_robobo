#!/usr/bin/env python3
"""
Learning Robobo Controller for Task 3: Object Pushing
======================================================

Main controller script for running Task 3 (Object Pushing to Target Location) with different
RL methods and modes. This script provides a simple interface for training and
evaluating RL agents for object pushing tasks.

Usage:
    python learning_robobo_controller.py --simulation [--method dqn] [--episodes 100]
    python learning_robobo_controller.py --hardware [--method dqn] [--episodes 50]
"""

import sys
import argparse

from learning_machines.test_actions import SimulationRobobo, HardwareRobobo
from learning_machines import (
    run_all_actions,
    object_pushing_task3,
    test_task3_capabilities,
    demo_task3_object_pushing
)


def main():
    parser = argparse.ArgumentParser(
        description='Learning Robobo Controller for Task 3: Object Pushing'
    )
    
    # Platform selection (required)
    platform_group = parser.add_mutually_exclusive_group(required=True)
    platform_group.add_argument('--hardware', action='store_true',
                               help='Run on hardware robot')
    platform_group.add_argument('--simulation', action='store_true',
                               help='Run in simulation')
    
    # RL method selection
    parser.add_argument('--method', choices=['dqn', 'qlearning', 'policy_gradient', 'actor_critic'],
                       default='dqn', help='RL method to use (default: dqn)')
    
    # Training parameters
    parser.add_argument('--episodes', type=int, default=100,
                       help='Number of training episodes (default: 100)')
    parser.add_argument('--mode', choices=['train', 'evaluate', 'train_and_evaluate'],
                       default='train', help='Training mode (default: train)')
    
    # Task selection
    parser.add_argument('--task', choices=['object_pushing', 'test_capabilities', 'demo'],
                       default='object_pushing', help='Task to run (default: object_pushing)')
    
    args = parser.parse_args()
    
    # Create robot interface
    if args.hardware:
        print("🤖 Creating hardware robot interface...")
        rob = HardwareRobobo(camera=True)
    else:
        print("🎮 Creating simulation robot interface...")
        rob = SimulationRobobo()
    
    try:
        print(f"🎯 Running Task 3: Object Pushing")
        print(f"Platform: {'Hardware' if args.hardware else 'Simulation'}")
        print(f"Method: {args.method.upper()}")
        print(f"Episodes: {args.episodes}")
        print(f"Mode: {args.mode}")
        print("="*60)
        
        if args.task == 'object_pushing':
            # Main object pushing task
            results = run_all_actions(
                rob=rob,
                rl_agent_type=args.method,
                rl_mode=args.mode,
                rl_episodes=args.episodes
            )
            
            print(f"\n✅ Task 3 completed successfully!")
            if 'success_rate' in results:
                print(f"Success Rate: {results.get('success_rate', 0):.1%}")
            
        elif args.task == 'test_capabilities':
            # Test system capabilities
            test_task3_capabilities(rob)
            
        elif args.task == 'demo':
            # Quick demo
            demo_task3_object_pushing(rob)
            
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        raise
    finally:
        # Cleanup
        if hasattr(rob, 'stop_simulation'):
            rob.stop_simulation()
        print("🧹 Cleanup completed")


if __name__ == "__main__":
    main()
