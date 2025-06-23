#!/usr/bin/env python3
"""
IR Sensor Monitor for Task 2: Green Food Collection
===================================================

Monitor IR sensor values in real-time to help with debugging robot navigation
and obstacle avoidance during food collection tasks.

This utility displays:
- Real-time IR sensor readings
- Sensor order: [BackL, BackR, FrontL, FrontR, FrontC, FrontRR, BackC, FrontLL]
- Obstacle detection thresholds
- Collision warnings

Usage:
    python3 monitor_ir_sensors.py --simulation --duration 30
    python3 monitor_ir_sensors.py --hardware --duration 60 --interval 1.0
"""

import argparse
import time
import sys

# Add the learning_machines package to the path
sys.path.append('/root/catkin_ws/src/learning_machines/src')

from learning_machines.test_actions import HardwareRobobo, SimulationRobobo

def monitor_ir_sensors(rob, duration=30, interval=0.5):
    """
    Monitor IR sensor values for Task 2 debugging
    
    Args:
        rob: Robot interface (hardware or simulation)
        duration: Duration in seconds to monitor
        interval: Time between readings in seconds
    """
    print("🔍 Starting IR sensor monitoring for Task 2...")
    print("Sensor Order: [BackL, BackR, FrontL, FrontR, FrontC, FrontRR, BackC, FrontLL]")
    print("=" * 80)
    
    # Task 2 specific thresholds
    if hasattr(rob, 'is_simulation') or 'Simulation' in str(type(rob)):
        collision_threshold = 0.05  # Simulation threshold
        platform = "Simulation"
    else:
        collision_threshold = 20.0  # Hardware threshold  
        platform = "Hardware"
        
    print(f"Platform: {platform}")
    print(f"Collision Threshold: {collision_threshold}")
    print(f"Monitoring Duration: {duration}s, Interval: {interval}s")
    print("-" * 80)
    
    start_time = time.time()
    end_time = start_time + duration
    reading_count = 0
    
    try:
        while time.time() < end_time:
            reading_count += 1
            ir_values = rob.read_irs()
            
            # Check for potential collisions
            min_value = min(ir_values)
            collision_warning = "⚠️  COLLISION RISK" if min_value < collision_threshold else "✅ Clear"
            
            # Format reading
            timestamp = time.time() - start_time
            print(f"[{timestamp:6.1f}s] #{reading_count:3d} | {ir_values} | {collision_warning}")
            
            # Detailed sensor analysis every 10 readings
            if reading_count % 10 == 0:
                sensor_names = ['BackL', 'BackR', 'FrontL', 'FrontR', 'FrontC', 'FrontRR', 'BackC', 'FrontLL']
                print("  Detailed Analysis:")
                for i, (name, value) in enumerate(zip(sensor_names, ir_values)):
                    status = "🔴 CLOSE" if value < collision_threshold else "🟢 OK"
                    print(f"    {name:8s}: {value:8.3f} {status}")
                print("-" * 80)
            
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print(f"\n⚠️  Monitoring interrupted by user after {reading_count} readings")
        
    print(f"\n✅ Monitoring complete - {reading_count} readings collected")
    print(f"Total duration: {time.time() - start_time:.1f}s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Monitor IR Sensor Values')
    parser.add_argument('--hardware', action='store_true', help='Run on hardware')
    parser.add_argument('--simulation', action='store_true', help='Run on simulation')
    parser.add_argument('--duration', type=int, default=30, help='Duration in seconds to monitor')
    parser.add_argument('--interval', type=float, default=0.5, help='Time between readings in seconds')
    
    args = parser.parse_args()
    
    if not (args.hardware or args.simulation):
        print("Error: Please specify either --hardware or --simulation")
        exit(1)
    
    if args.hardware:
        print("Monitoring IR sensors on hardware...")
        rob = HardwareRobobo()
    else:
        print("Monitoring IR sensors on simulation...")
        rob = SimulationRobobo()
        rob.play_simulation()
    
    try:
        monitor_ir_sensors(rob, args.duration, args.interval)
    finally:
        if isinstance(rob, SimulationRobobo):
            rob.stop_simulation()
