#!/usr/bin/env python3
import sys
import argparse

from learning_machines.test_actions import SimulationRobobo, HardwareRobobo
from learning_machines import obstacle_avoidance_task1, wall_following_algorithm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Task 0: Obstacle Avoidance')
    parser.add_argument('--hardware', action='store_true', help='Run on hardware')
    parser.add_argument('--simulation', action='store_true', help='Run on simulation')
    parser.add_argument('--iterations', type=int, default=100, help='Number of control iterations')
    parser.add_argument('--duration', type=int, default=60, help='Duration in seconds for walk_until_obstacle')
    parser.add_argument('--no-save', action='store_true', help='Don\'t save data')
    parser.add_argument('--method', choices=['obstacle_avoidance_task1', 'wall_following_algorithm'], 
                        default='obstacle_avoidance_task1', 
                        help='Choose which method to run')
    
    args = parser.parse_args()
    
    if not (args.hardware or args.simulation):
        print("Error: Please specify either --hardware or --simulation")
        sys.exit(1)
    
    if args.hardware:
        print("Running Task 0 on hardware...")
        rob = HardwareRobobo(camera=True)
    else:
        print("Running Task 0 on simulation...")
        rob = SimulationRobobo()
    
    # Run the selected method
    if args.method == 'obstacle_avoidance_task1':
        print("Using obstacle_avoidance_task1 method")
        data = obstacle_avoidance_task1(
            rob, 
            duration_seconds=args.duration, 
            save_data=not args.no_save
        )
    else:
        print("Using wall_following_algorithm method")
        wall_following_algorithm(
            rob,
            duration_seconds=args.duration
        )
    
    print("Task 0 completed!")
