#!/usr/bin/env python3
"""
RL Controller for Task 2 - Green Food Collection
Demonstrates all 4 reinforcement learning approaches for food collection using computer vision
"""

import sys
import argparse
from pathlib import Path

from robobo_interface import SimulationRobobo, HardwareRobobo
from test_actions import green_food_collection_task2

def main():
    parser = argparse.ArgumentParser(description='RL-based Task 2: Green Food Collection')
    parser.add_argument('--hardware', action='store_true', help='Run on hardware')
    parser.add_argument('--simulation', action='store_true', help='Run on simulation (default)')
    
    # RL-specific arguments
    parser.add_argument('--rl', action='store_true', help='Use reinforcement learning')
    parser.add_argument('--agent', choices=['qlearning', 'dqn', 'policy_gradient', 'actor_critic'], 
                        default='dqn', help='RL agent type (default: dqn)')
    parser.add_argument('--mode', choices=['train', 'evaluate', 'train_and_evaluate'], 
                        default='train', help='RL mode')
    parser.add_argument('--episodes', type=int, default=100, help='Number of episodes')
    parser.add_argument('--model-path', type=str, help='Path to saved model for evaluation')
    
    # Demo mode - run all 4 RL methods
    parser.add_argument('--demo-all', action='store_true', 
                        help='Demo all 4 RL methods (short training)')
    
    args = parser.parse_args()
    
    # Default to simulation if neither specified
    if not (args.hardware or args.simulation):
        args.simulation = True
    
    if args.hardware and args.simulation:
        print("Error: Please specify either --hardware or --simulation, not both")
        sys.exit(1)
    
    # Create robot interface
    if args.hardware:
        print("Running on hardware...")
        rob = HardwareRobobo(camera=True)
    else:
        print("Running on simulation...")
        rob = SimulationRobobo()
    
    try:
        if args.demo_all:
            # Demo all 4 RL methods with short training
            print("\n" + "="*80)
            print("DEMO: ALL 4 REINFORCEMENT LEARNING METHODS - TASK 2: GREEN FOOD COLLECTION")
            print("="*80)
            
            methods = ['qlearning', 'dqn', 'policy_gradient', 'actor_critic']
            demo_episodes = 20  # Short demo episodes
            
            for i, method in enumerate(methods, 1):
                print(f"\n[{i}/4] Training {method.upper()} agent for food collection...")
                
                try:
                    results = green_food_collection_task2(
                        rob, 
                        agent_type=method, 
                        mode='train',
                        num_episodes=demo_episodes
                    )
                    print(f"{method.upper()} completed successfully!")
                    print(f"Results: {results.get('success_rate', 0):.1%} success rate")
                    
                except Exception as e:
                    print(f"Error training {method}: {e}")
                    continue
                
                # Small pause between methods
                import time
                time.sleep(2)
            
            print("\n" + "="*80)
            print("DEMO COMPLETED! Check the results/ folder for saved models and plots.")
            print("="*80)
        
        elif args.rl:
            # Run single RL method for food collection
            print(f"\nRunning RL-based food collection with {args.agent.upper()}")
            
            results = green_food_collection_task2(
                rob,
                agent_type=args.agent,
                mode=args.mode,
                num_episodes=args.episodes
            )
            
            print(f"\nRL training/evaluation completed!")
            print(f"Final results: {results}")
            
        else:
            # Run demo without RL (basic food collection test)
            print("\nRunning basic food collection test (no RL)")
            from test_actions import test_task2_capabilities
            test_task2_capabilities(rob)
            
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        raise
    finally:
        # Clean up
        if hasattr(rob, 'stop_simulation'):
            rob.stop_simulation()
        print("Cleanup completed")

if __name__ == "__main__":
    main()
