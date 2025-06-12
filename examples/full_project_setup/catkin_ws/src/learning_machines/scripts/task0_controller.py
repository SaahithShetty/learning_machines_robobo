#!/usr/bin/env python3
import sys
import argparse

from learning_machines.test_actions import SimulationRobobo, HardwareRobobo
from learning_machines import obstacle_avoidance_task1, wall_following_algorithm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Baseline Obstacle Avoidance Algorithms')
    parser.add_argument('--hardware', action='store_true', help='Run on hardware')
    parser.add_argument('--simulation', action='store_true', help='Run on simulation')
    parser.add_argument('--duration', type=int, default=60, help='Duration in seconds to run algorithm')
    parser.add_argument('--max-distance', type=float, default=5.0, help='Maximum distance to travel (safety limit)')
    parser.add_argument('--wall-distance', type=float, default=0.3, help='Target wall distance for wall-following')
    parser.add_argument('--threshold', type=float, help='Custom obstacle detection threshold')
    parser.add_argument('--no-save', action='store_true', help='Don\'t save data')
    parser.add_argument('--method', choices=['obstacle_avoidance_task1', 'wall_following_algorithm'], 
                        default='obstacle_avoidance_task1', 
                        help='Choose which baseline method to run')
    
    args = parser.parse_args()
    
    if not (args.hardware or args.simulation):
        print("Error: Please specify either --hardware or --simulation")
        sys.exit(1)
    
    if args.hardware:
        print("Running baseline algorithm on hardware...")
        rob = HardwareRobobo(camera=True)
        platform = "hardware"
    else:
        print("Running baseline algorithm in simulation...")
        rob = SimulationRobobo()
        platform = "simulation"
    
    print(f"="*60)
    print(f"BASELINE ALGORITHM: {args.method.upper().replace('_', ' ')}")
    print(f"Platform: {platform}")
    print(f"Duration: {args.duration}s")
    print(f"="*60)
    
    # Run the selected method
    if args.method == 'obstacle_avoidance_task1':
        print("Running advanced obstacle avoidance algorithm...")
        data = obstacle_avoidance_task1(
            rob, 
            max_distance=args.max_distance,
            duration_seconds=args.duration, 
            save_data=not args.no_save,
            threshold=args.threshold
        )
        print(f"Baseline obstacle avoidance completed!")
        if data and isinstance(data, dict) and 'metrics' in data:
            metrics = data['metrics']
            print(f"Distance traveled: {metrics['total_distance']:.2f}m")
            print(f"Collision events: {metrics['collision_events']}")
            print(f"Efficiency score: {metrics['efficiency_score']:.3f}")
    else:
        print("Running wall-following algorithm...")
        wall_following_algorithm(
            rob,
            duration_seconds=args.duration,
            wall_distance=args.wall_distance
        )
        print(f"Baseline wall-following completed!")
    
    print(f"Baseline algorithm execution finished!")
