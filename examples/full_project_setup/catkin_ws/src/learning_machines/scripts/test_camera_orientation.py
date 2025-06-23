#!/usr/bin/env python3
"""
Quick test script to verify camera orientation fix
Tests the new tilt angle to ensure camera faces forward toward food
"""

import time
import sys
import os

# Add the learning_machines package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from learning_machines.test_actions import RoboboEnv

def test_camera_orientation():
    """Test that camera is now properly oriented forward"""
    
    print("🎥 Testing Camera Orientation Fix")
    print("=" * 50)
    
    # Initialize environment
    env = RoboboEnv()
    rob = env.robot
    
    print(f"Environment type: {'Simulation' if env.is_simulation else 'Hardware'}")
    
    # Get initial camera position
    initial_pan = rob.read_phone_pan()
    initial_tilt = rob.read_phone_tilt()
    print(f"Initial Camera Position - Pan: {initial_pan}, Tilt: {initial_tilt}")
    
    # Initialize camera with new settings
    print("\n🔧 Applying camera orientation fix...")
    env._initialize_camera()
    time.sleep(1)
    
    # Check new position
    new_pan = rob.read_phone_pan()
    new_tilt = rob.read_phone_tilt()
    print(f"New Camera Position - Pan: {new_pan}, Tilt: {new_tilt}")
    
    # Test food detection with new orientation
    print("\n🍎 Testing food detection with corrected orientation...")
    
    detection_results = []
    for i in range(10):
        food_count = env._count_food_in_image()
        detection_results.append(food_count)
        print(f"Frame {i+1}: Detected {food_count} food items")
        time.sleep(0.2)
    
    # Analysis
    avg_detection = sum(detection_results) / len(detection_results)
    successful_detections = sum(1 for x in detection_results if x > 0)
    
    print(f"\n📊 Detection Results:")
    print(f"   Average food detected per frame: {avg_detection:.1f}")
    print(f"   Frames with food detected: {successful_detections}/{len(detection_results)}")
    print(f"   Detection success rate: {(successful_detections/len(detection_results)*100):.1f}%")
    
    # Visual test - take a snapshot
    print(f"\n📸 Taking test image...")
    try:
        image = rob.read_image_front()
        if image is not None:
            print(f"   ✅ Successfully captured image (shape: {image.shape})")
            # Save image for manual inspection
            import cv2
            cv2.imwrite('/tmp/camera_test_image.jpg', image)
            print(f"   💾 Test image saved to /tmp/camera_test_image.jpg")
        else:
            print(f"   ❌ Failed to capture image")
    except Exception as e:
        print(f"   ❌ Image capture error: {e}")
    
    # Test some basic movements to see if food detection improves
    print(f"\n🤖 Testing food detection during movement...")
    
    # Test food detection with fixed camera position (no pan movements)
    print(f"\n🔍 Testing food detection with default camera position:")
    
    # Single test at default position
    food_count = env._count_food_in_image()
    current_pan = rob.read_phone_pan()
    current_tilt = rob.read_phone_tilt()
    print(f"   Pan: {current_pan}, Tilt: {current_tilt}, Food detected: {food_count}")
    
    print(f"\n✅ Camera orientation test complete!")
    print(f"💡 Camera now uses default orientation only - no pan/tilt control")

if __name__ == "__main__":
    test_camera_orientation()
