#!/usr/bin/env python3
"""
Calibration Arena Diagnostic Tool
Helps verify sensor functionality and scene setup before running full calibration
"""

import time
import json
from pathlib import Path
from robobo_interface import SimulationRobobo

def test_sensor_connectivity():
    """Test if robot and sensors are properly connected"""
    print("🔍 SENSOR CONNECTIVITY TEST")
    print("=" * 40)
    
    try:
        robot = SimulationRobobo()
        print("✅ Robot connection established")
        
        if not robot.is_running():
            print("▶️  Starting simulation...")
            robot.play_simulation()
            time.sleep(2.0)
        
        print("✅ Simulation is running")
        
        # Test IR sensor readings
        print("\n📡 Testing IR sensor readings...")
        ir_data = robot.read_irs()
        print(f"Raw IR data: {ir_data}")
        
        # Check if sensors are functional
        non_zero_sensors = [i for i, val in enumerate(ir_data) if val is not None and val > 0]
        
        if not non_zero_sensors:
            print("❌ All sensors returning 0.0 or None")
            print("💡 Possible issues:")
            print("   - Robot not properly loaded in scene")
            print("   - IR sensors not configured correctly")
            print("   - No obstacles within sensor range")
            print("   - Simulation not running")
        else:
            print(f"✅ Active sensors detected: {non_zero_sensors}")
            
        # Test robot position (simulation only)
        pos = robot.get_position()
        orient = robot.get_orientation()
        print(f"\n📍 Robot position: {pos}")
        print(f"🧭 Robot orientation: {orient}")
        
        return True
        
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        print("💡 Make sure:")
        print("   - CoppeliaSim is running")
        print("   - Calibration scene is loaded")
        print("   - API server is accessible (port 23000)")
        return False

def analyze_calibration_data():
    """Analyze existing calibration data if available"""
    calibration_file = Path("/root/results/ir_sensor_calibration_data.json")
    
    if not calibration_file.exists():
        print("📄 No existing calibration data found")
        return
    
    print("\n📊 EXISTING CALIBRATION DATA ANALYSIS")
    print("=" * 40)
    
    with open(calibration_file, 'r') as f:
        data = json.load(f)
    
    hw_data = data.get("hardware", {})
    sim_data = data.get("simulation", {})
    
    if hw_data:
        print("🤖 Hardware calibration data found:")
        for distance, sensors in hw_data.items():
            front_sensor_means = []
            for sensor_idx in [4, 5, 7]:  # Front sensors
                if str(sensor_idx) in sensors and sensors[str(sensor_idx)].get("mean") is not None:
                    front_sensor_means.append(sensors[str(sensor_idx)]["mean"])
            
            if front_sensor_means:
                avg = sum(front_sensor_means) / len(front_sensor_means)
                print(f"   {distance}cm: {avg:.2f} (avg front sensors)")
            else:
                print(f"   {distance}cm: No valid readings")
    
    if sim_data:
        print("\n🎮 Simulation calibration data found:")
        for distance, sensors in sim_data.items():
            front_sensor_means = []
            for sensor_idx in [4, 5, 7]:  # Front sensors
                if str(sensor_idx) in sensors and sensors[str(sensor_idx)].get("mean") is not None:
                    front_sensor_means.append(sensors[str(sensor_idx)]["mean"])
            
            if front_sensor_means:
                avg = sum(front_sensor_means) / len(front_sensor_means)
                print(f"   {distance}cm: {avg:.2f} (avg front sensors)")
            else:
                print(f"   {distance}cm: No valid readings")
    
    if not hw_data and not sim_data:
        print("📊 No calibration data available yet")

def provide_setup_recommendations():
    """Provide recommendations for optimal calibration setup"""
    print("\n💡 CALIBRATION SETUP RECOMMENDATIONS")
    print("=" * 40)
    print("1. 🏗️  Scene Setup:")
    print("   - Use a large, flat floor (5m x 5m)")
    print("   - Add a smooth, non-reflective wall")
    print("   - Place distance markers at 5cm intervals")
    print("   - Ensure uniform lighting (no shadows)")
    
    print("\n2. 🤖 Robot Positioning:")
    print("   - Start robot ~40cm from wall")
    print("   - Ensure robot faces wall perpendicularly")
    print("   - Use CoppeliaSim tools for precise positioning")
    print("   - Front IR sensors should point toward wall")
    
    print("\n3. 📏 Distance Measurements:")
    print("   - Measure from robot front edge to wall")
    print("   - Use precise positioning: 5, 10, 15, 20, 25, 30cm")
    print("   - Take readings with robot completely stationary")
    print("   - Verify readings are non-zero and reasonable")
    
    print("\n4. 🎯 Expected Simulation Readings:")
    print("   - 5cm: ~1800-1900 (very close)")
    print("   - 10cm: ~1600-1700 (close)")
    print("   - 15cm: ~1400-1500 (medium)")
    print("   - 20cm: ~1200-1300 (medium-far)")
    print("   - 25cm: ~1000-1100 (far)")
    print("   - 30cm: ~800-900 (maximum range)")

def main():
    """Main diagnostic interface"""
    print("🔧 CALIBRATION ARENA DIAGNOSTIC TOOL")
    print("=" * 50)
    print("This tool helps verify your calibration setup before running the full calibration.")
    print()
    
    # Test basic connectivity
    if test_sensor_connectivity():
        print("\n🎉 Basic connectivity test passed!")
    else:
        print("\n❌ Basic connectivity test failed!")
        print("Please fix connection issues before proceeding with calibration.")
        return
    
    # Analyze existing data
    analyze_calibration_data()
    
    # Provide recommendations
    provide_setup_recommendations()
    
    print("\n🚀 Ready to run full calibration!")
    print("Use: python3 /root/catkin_ws/src/learning_machines/src/learning_machines/sensor_calibration_tool.py")

if __name__ == "__main__":
    main()
