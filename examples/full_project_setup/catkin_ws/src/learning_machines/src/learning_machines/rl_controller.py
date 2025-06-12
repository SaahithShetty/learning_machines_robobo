#!/usr/bin/env python3
"""
RL Controller for Task 1 - Obstacle Avoidance
Demonstrates all 4 reinforcement learning approaches
"""

import sys
import argparse
from pathlib import Path

from robobo_interface import SimulationRobobo, HardwareRobobo
from learning_machines import run_all_actions

def main():
    parser = argparse.ArgumentParser(description='RL-based Task 1: Obstacle Avoidance')
    parser.add_argument('--hardware', action='store_true', help='Run on hardware')
    parser.add_argument('--simulation', action='store_true', help='Run on simulation (default)')
    
    # RL-specific arguments
    parser.add_argument('--rl', action='store_true', help='Use reinforcement learning')
    parser.add_argument('--agent', choices=['qlearning', 'dqn', 'policy_gradient', 'actor_critic'], 
                        default='qlearning', help='RL agent type')
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
            print("DEMO: ALL 4 REINFORCEMENT LEARNING METHODS")
            print("="*80)
            
            methods = ['qlearning', 'dqn', 'policy_gradient', 'actor_critic']
            demo_episodes = 20  # Short demo episodes
            
            for i, method in enumerate(methods, 1):
                print(f"\n[{i}/4] Training {method.upper()} agent...")
                
                try:
                    results = run_all_actions(
                        rob, 
                        use_rl=True, 
                        rl_agent_type=method, 
                        rl_mode='train_and_evaluate',
                        rl_episodes=demo_episodes
                    )
                    print(f"{method.upper()} completed successfully!")
                    
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
            # Run single RL method
            print(f"\nRunning RL-based obstacle avoidance with {args.agent.upper()}")
            
            results = run_all_actions(
                rob,
                use_rl=True,
                rl_agent_type=args.agent,
                rl_mode=args.mode,
                rl_episodes=args.episodes
            )
            
            print(f"\nRL training/evaluation completed!")
            
        else:
            # Run original rule-based approach
            print("\nRunning rule-based obstacle avoidance")
            run_all_actions(rob, use_rl=False)
            
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
