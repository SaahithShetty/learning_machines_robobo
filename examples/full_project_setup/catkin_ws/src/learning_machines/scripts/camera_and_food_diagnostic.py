#!/usr/bin/env python3
"""
Camera and Food Collection Diagnostic Tool
==========================================

This script helps diagnose and verify:
1. Camera orientation and positioning
2. Food detection accuracy 
3. Food collection tracking consistency
4. Simulation vs Python collision detection alignment

Usage:
    python3 camera_and_food_diagnostic.py --simulation --duration 60
    python3 camera_and_food_diagnostic.py --hardware --duration 120
"""

import argparse
import time
import sys
import cv2
import numpy as np

# Add the learning_machines package to the path
sys.path.append('/root/catkin_ws/src/learning_machines/src')

from learning_machines.test_actions import (
    HardwareRobobo, SimulationRobobo, 
    FoodVisionProcessor, RobotEnvironment
)

def test_camera_and_food_system(rob, duration=60):
    """Test camera positioning, food detection, and collection tracking"""
    
    print("🔍 Camera and Food Collection Diagnostic")
    print("="*50)
    
    # Determine environment type
    env_type = "simulation" if isinstance(rob, SimulationRobobo) else "hardware"
    print(f"Environment: {env_type}")
    
    # Initialize vision processor
    vision_processor = FoodVisionProcessor(environment_type=env_type)
    print(f"Vision ranges: {vision_processor.green_ranges}")
    
    # Initialize robot environment for testing
    environment = RobotEnvironment(rob, vision_processor, max_episode_time=duration)
    
    if isinstance(rob, SimulationRobobo):
        rob.play_simulation()
        print("Simulation started")
    
    # Test 1: Camera Positioning and Field of View
    print(f"\n📹 Camera Positioning Test")
    print("-" * 30)
    
    try:
        current_pan = rob.read_phone_pan()
        current_tilt = rob.read_phone_tilt()
        print(f"Initial Camera Position - Pan: {current_pan}, Tilt: {current_tilt}")
        
        # Test camera initialization
        environment._initialize_camera()
        time.sleep(1)
        
        new_pan = rob.read_phone_pan()
        new_tilt = rob.read_phone_tilt()
        print(f"After Initialization - Pan: {new_pan}, Tilt: {new_tilt}")
        
    except Exception as e:
        print(f"❌ Camera positioning error: {e}")
    
    # Test 2: Food Detection Accuracy
    print(f"\n🍎 Food Detection Test")
    print("-" * 30)
    
    detection_count = 0
    total_frames = 20
    
    for i in range(total_frames):
        try:
            # Get camera frame
            camera_frame = rob.read_image_front()
            if camera_frame is None:
                print(f"  Frame {i+1}: No camera data")
                continue
            
            # Detect food
            food_objects, mask = vision_processor.detect_green_food(camera_frame)
            
            if food_objects:
                detection_count += 1
                best_food = food_objects[0]  # Most confident detection
                print(f"  Frame {i+1}: {len(food_objects)} food(s) detected")
                print(f"    Best: Distance={best_food['distance']:.2f}m, "
                      f"Angle={best_food['angle']:.1f}°, "
                      f"Confidence={best_food['confidence']:.2f}")
            else:
                print(f"  Frame {i+1}: No food detected")
            
            # Save debug images occasionally
            if i % 5 == 0:
                cv2.imwrite(f"/tmp/diagnostic_frame_{i}.png", camera_frame)
                cv2.imwrite(f"/tmp/diagnostic_mask_{i}.png", mask)
            
            time.sleep(0.5)
            
        except Exception as e:
            print(f"  Frame {i+1}: Error - {e}")
    
    detection_rate = (detection_count / total_frames) * 100
    print(f"\n📊 Detection Summary:")
    print(f"  Detection rate: {detection_rate:.1f}% ({detection_count}/{total_frames} frames)")
    
    # Test 3: Food Collection Consistency
    print(f"\n🎯 Food Collection Tracking Test")
    print("-" * 30)
    
    # Test collision detection without movement
    for test_num in range(5):
        print(f"\nTest {test_num + 1}:")
        
        try:
            # Get current IR readings
            ir_values = rob.read_irs()
            if ir_values and len(ir_values) >= 5:
                front_sensors = [ir_values[2], ir_values[3], ir_values[4]]
                valid_sensors = [val for val in front_sensors if val is not None]
                if valid_sensors:
                    min_front = min(valid_sensors)
                    print(f"  IR Front Min: {min_front:.1f}")
                    
                    # Test collision detection logic
                    collision_detected = environment._check_food_collision()
                    print(f"  Python Collision Detection: {collision_detected}")
                    
                    # Get simulation count if available
                    if hasattr(rob, 'get_nr_food_collected'):
                        try:
                            sim_count = rob.get_nr_food_collected()
                            print(f"  Simulation Food Count: {sim_count}")
                        except Exception as e:
                            print(f"  Simulation count error: {e}")
                    
                    # Test sync function
                    sync_result = environment._sync_food_count_with_simulation()
                    print(f"  Sync Result: {sync_result}")
                    print(f"  Python Food Count: {environment.food_collected}")
                else:
                    print("  No valid IR sensor readings")
            else:
                print("  No IR data available")
                
        except Exception as e:
            print(f"  Test {test_num + 1} error: {e}")
        
        time.sleep(2)
    
    # Test 4: Panoramic Scan
    print(f"\n🔄 Panoramic Scan Test")
    print("-" * 30)
    
    try:
        original_pan = rob.read_phone_pan()
        print(f"Original pan position: {original_pan}")
        
        # Test panoramic food detection
        vision_data = environment._get_panoramic_food_state()
        print(f"Panoramic scan result: {vision_data}")
        
        final_pan = rob.read_phone_pan()
        print(f"Final pan position: {final_pan}")
        
        if abs(final_pan - original_pan) > 5:
            print(f"⚠️  Camera not properly reset! (diff: {abs(final_pan - original_pan)})")
        else:
            print("✅ Camera properly reset after scan")
            
    except Exception as e:
        print(f"❌ Panoramic scan error: {e}")
    
    # Summary
    print(f"\n🏁 Diagnostic Summary")
    print("="*50)
    print(f"Environment Type: {env_type}")
    print(f"Camera System: {'✅ Working' if detection_count > 0 else '❌ Issues detected'}")
    print(f"Food Detection Rate: {detection_rate:.1f}%")
    print(f"Food Collection Logic: {'✅ Implemented' if hasattr(environment, '_check_food_collision') else '❌ Missing'}")
    
    if isinstance(rob, SimulationRobobo):
        rob.stop_simulation()
        print("Simulation stopped")
    
    print(f"\n📁 Debug images saved to /tmp/diagnostic_*.png")

def main():
    parser = argparse.ArgumentParser(description='Camera and Food Collection Diagnostic')
    parser.add_argument('--simulation', action='store_true', help='Use simulation robot')
    parser.add_argument('--hardware', action='store_true', help='Use hardware robot')
    parser.add_argument('--duration', type=int, default=60, help='Test duration in seconds')
    
    args = parser.parse_args()
    
    if args.simulation:
        print("Connecting to simulation...")
        rob = SimulationRobobo(api_port=19999)
    elif args.hardware:
        print("Connecting to hardware...")
        rob = HardwareRobobo(camera=True)
    else:
        print("Please specify --simulation or --hardware")
        return
    
    try:
        test_camera_and_food_system(rob, args.duration)
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n🔚 Diagnostic complete")

if __name__ == "__main__":
    main()
