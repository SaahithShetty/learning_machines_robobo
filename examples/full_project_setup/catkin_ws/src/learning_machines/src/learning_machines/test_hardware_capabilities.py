#!/usr/bin/env python3
"""
Hardware Capability Test Script

This script tests all Task 2 capabilities on the hardware robot.
Run this before training to ensure everything is working properly.
"""

import sys
import time

# Add the learning_machines package to the path
sys.path.append('/root/catkin_ws/src/learning_machines/src')

from learning_machines.test_actions import HardwareRobobo, test_task2_capabilities

def main():
    print("=" * 60)
    print("HARDWARE CAPABILITY TEST FOR TASK 2")
    print("=" * 60)
    
    try:
        # Connect to hardware robot with camera enabled
        print("Connecting to hardware robot...")
        rob = HardwareRobobo(camera=True)
        
        # Wait for connection
        print("Waiting for robot connection...")
        time.sleep(3)
        
        print("✅ Connected successfully!")
        
        # Run capability tests
        test_task2_capabilities(rob)
        
        print("\n" + "=" * 60)
        print("CAPABILITY TEST COMPLETED")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Error during capability test: {e}")
        print("Make sure:")
        print("  1. Robot is powered on and connected")
        print("  2. ROS is running properly")
        print("  3. Camera permissions are set")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
