#!/usr/bin/env python3
"""
Simple camera orientation test that can be run immediately
"""
import sys
import os
sys.path.insert(0, '/Users/saahithshetty/Documents/Coding/Learning_Machines/learning_machines_robobo/examples/full_project_setup/catkin_ws/src')

def test_camera_fix():
    print("🎥 Testing Camera Orientation Fix")
    print("=" * 40)
    
    try:
        from learning_machines.test_actions import RoboboEnv
        
        # Initialize environment
        env = RoboboEnv()
        rob = env.robot
        
        print(f"Environment: {'Simulation' if env.is_simulation else 'Hardware'}")
        
        # Get current camera position
        print(f"Current Camera - Pan: {rob.read_phone_pan()}, Tilt: {rob.read_phone_tilt()}")
        
        # Apply camera fix
        print("🔧 Applying camera orientation fix...")
        env._initialize_camera()
        
        # Check new position
        new_pan = rob.read_phone_pan()
        new_tilt = rob.read_phone_tilt()
        print(f"Fixed Camera - Pan: {new_pan}, Tilt: {new_tilt}")
        
        # Quick food detection test
        print("🍎 Testing food detection...")
        food_count = env._count_food_in_image()
        print(f"Food detected: {food_count}")
        
        if env.is_simulation:
            expected_tilt = -30
            if abs(new_tilt - expected_tilt) < 5:
                print("✅ Camera tilt correctly set to forward-facing angle")
            else:
                print(f"⚠️  Camera tilt {new_tilt} differs from expected {expected_tilt}")
        
        print("🎯 Test complete!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_camera_fix()
