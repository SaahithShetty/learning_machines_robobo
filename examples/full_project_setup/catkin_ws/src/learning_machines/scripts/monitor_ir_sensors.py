#!/usr/bin/env python3
import argparse
import time

from learning_machines.test_actions import HardwareRobobo, SimulationRobobo

def monitor_ir_sensors(rob, duration=30, interval=0.5):
    """
    Monitor IR sensor values for a specified duration
    
    Args:
        rob: Robot interface (hardware or simulation)
        duration: Duration in seconds to monitor
        interval: Time between readings in seconds
    """
    print("Starting IR sensor monitoring...")
    print("[BackL, BackR, FrontL, FrontR, FrontC, FrontRR, BackC, FrontLL]")
    print("--------------------------------------------------------------")
    
    start_time = time.time()
    end_time = start_time + duration
    
    while time.time() < end_time:
        ir_values = rob.read_irs()
        print(f"IR values: {ir_values}")
        time.sleep(interval)
    
    print("Monitoring complete")

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
