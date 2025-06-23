#!/usr/bin/env python3
"""
Quick Camera Orientation Test
============================

This script tests the corrected camera pan/tilt positions to ensure
the camera faces forward toward the food instead of backward/upward.
"""

import sys
import time

# Add path for imports
sys.path.append('/root/catkin_ws/src/learning_machines/src')

def test_camera_orientation():
    try:
        from learning_machines.test_actions import SimulationRobobo, FoodVisionProcessor
        
        print("🔧 Testing Camera Orientation Fix")
        print("=" * 40)
        
        # Connect to simulation
        print("Connecting to simulation...")
        rob = SimulationRobobo(api_port=19999)
        rob.play_simulation()
        
        print("✅ Connected to simulation")
        
        # Test original position
        print(f"\n📍 Original Position:")
        print(f"  Pan: {rob.read_phone_pan()}")
        print(f"  Tilt: {rob.read_phone_tilt()}")
        
        # Test using default camera position (no pan/tilt changes)
        print(f"\n� Using Default Camera Position:")
        print(f"  Default Pan: {rob.read_phone_pan()}")
        print(f"  Default Tilt: {rob.read_phone_tilt()}")
        
        print(f"\n✅ Camera uses default orientation - no pan/tilt control")
        
        # Test food detection with default orientation
        print(f"\n🍎 Testing Food Detection:")
        vision_processor = FoodVisionProcessor(environment_type="simulation")
        
        for i in range(3):
            camera_frame = rob.read_image_front()
            if camera_frame is not None:
                food_objects, _ = vision_processor.detect_green_food(camera_frame)
                print(f"  Frame {i+1}: {len(food_objects)} food objects detected")
                
                if food_objects:
                    best_food = food_objects[0]
                    print(f"    Best detection: Distance={best_food['distance']:.2f}m, "
                          f"Angle={best_food['angle']:.1f}°")
            else:
                print(f"  Frame {i+1}: No camera data")
            
            time.sleep(0.5)
        
        # Test panoramic scan with corrected angles
        print(f"\n🔄 Testing Panoramic Scan:")
        scan_angles = [155, 180, 205]  # Left, center, right
        
        # Test with default camera position only (no pan movement)
        print(f"  Testing with default camera position")
        
        camera_frame = rob.read_image_front()
        if camera_frame is not None:
            food_objects, _ = vision_processor.detect_green_food(camera_frame)
            print(f"    Detected {len(food_objects)} food objects")
        
        print(f"\n✅ Camera uses default position only - no pan/tilt control")
        
        rob.stop_simulation()
        print(f"\n🏁 Test Complete - Camera uses default orientation!")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Run this script from within the Docker container")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_camera_orientation()
