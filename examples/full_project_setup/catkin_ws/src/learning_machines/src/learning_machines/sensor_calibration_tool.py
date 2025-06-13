#!/usr/bin/env python3
"""
IR Sensor Calibration Tool for Robobo Robot
Collects distance-to-sensor-reading mappings for both simulation and hardware platforms
"""

import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime

from robobo_interface import IRobobo, SimulationRobobo, HardwareRobobo


class IRSensorCalibrator:
    """Calibration utility for IR sensors across simulation and hardware platforms"""
    
    def __init__(self):
        self.calibration_data = {
            "simulation": {},
            "hardware": {},
            "metadata": {
                "created": datetime.now().isoformat(),
                "distances_cm": [5, 10, 15, 20, 25, 30],
                "readings_per_distance": 10,
                "card_length_cm": 5,
                "measurement_method": "5cm increments",
                "sensor_layout": {
                    0: "BackL", 1: "BackR", 2: "FrontL", 3: "FrontR",
                    4: "FrontC", 5: "FrontRR", 6: "BackC", 7: "FrontLL"
                }
            }
        }
        
        # Focus on front center sensor for primary calibration
        self.primary_sensors = [4]  # FrontC only
        # Using 5cm increments: 5cm, 10cm, 15cm, 20cm, 25cm, 30cm
        self.distances_cm = [5, 10, 15, 20, 25, 30]
        self.readings_per_distance = 10
        
        # Results storage - save to /root/results folder which is mounted to the host
        self.calibration_file = Path("/root/results/ir_sensor_calibration_data.json")
    
    def collect_sensor_readings(self, robot: IRobobo, distance_cm: float, num_readings: int = 10) -> Dict:
        """Collect multiple IR sensor readings at a specific distance"""
        print(f"  📏 Collecting {num_readings} readings at {distance_cm}cm...")
        
        readings = {i: [] for i in range(8)}  # 8 IR sensors
        
        for reading_num in range(num_readings):
            ir_data = robot.read_irs()
            
            # Store each sensor reading
            for sensor_idx, value in enumerate(ir_data):
                if value is not None:
                    readings[sensor_idx].append(value)
            
            # Brief pause between readings
            time.sleep(0.2)
            
            # Progress indicator
            if (reading_num + 1) % 3 == 0:
                print(f"    📊 Reading {reading_num + 1}/{num_readings}")
        
        # Calculate statistics for each sensor
        sensor_stats = {}
        for sensor_idx in range(8):
            if readings[sensor_idx]:
                sensor_stats[sensor_idx] = {
                    "raw_readings": readings[sensor_idx],
                    "mean": np.mean(readings[sensor_idx]),
                    "std": np.std(readings[sensor_idx]),
                    "min": np.min(readings[sensor_idx]),
                    "max": np.max(readings[sensor_idx]),
                    "count": len(readings[sensor_idx])
                }
            else:
                sensor_stats[sensor_idx] = {
                    "raw_readings": [],
                    "mean": None,
                    "std": None,
                    "min": None,
                    "max": None,
                    "count": 0
                }
        
        return sensor_stats
    
    def calibrate_simulation(self, robot: SimulationRobobo):
        """Run calibration procedure for simulation robot"""
        print("🎮 Starting SIMULATION calibration...")
        print("📋 Instructions:")
        print("   1. Make sure CoppeliaSim is running with your arena scene")
        print("   2. Position robot in center of arena, facing a wall")
        print("   3. Use CoppeliaSim interface to move robot to exact distances")
        print("   4. Press Enter when robot is positioned correctly at each distance")
        print("")
        
        if not robot.is_running():
            print("▶️  Starting simulation...")
            robot.play_simulation()
            time.sleep(2.0)
        
        platform_data = {}
        
        for distance_cm in self.distances_cm:
            print(f"\n🎯 Distance: {distance_cm}cm")
            print(f"   Move robot to {distance_cm}cm from wall using CoppeliaSim")
            input(f"   Press Enter when robot is positioned at {distance_cm}cm: ")
            
            # Collect readings
            sensor_data = self.collect_sensor_readings(robot, distance_cm, self.readings_per_distance)
            platform_data[distance_cm] = sensor_data
            
            # Show primary sensor readings
            front_sensors_mean = []
            for sensor_idx in self.primary_sensors:
                if sensor_data[sensor_idx]["mean"] is not None:
                    front_sensors_mean.append(sensor_data[sensor_idx]["mean"])
            
            if front_sensors_mean:
                avg_front = np.mean(front_sensors_mean)
                print(f"   ✅ Average front sensor reading: {avg_front:.2f}")
            else:
                print("   ⚠️  No front sensor readings detected")
        
        self.calibration_data["simulation"] = platform_data
        print("\n✅ Simulation calibration complete!")
    
    def calibrate_hardware(self, robot: HardwareRobobo):
        """Run calibration procedure for hardware robot"""
        print("🤖 Starting HARDWARE calibration...")
        print("📋 Instructions:")
        print("   1. Place robot on flat surface facing a smooth wall")
        print("   2. Use a ruler or measuring tape to measure exact distances:")
        print("      • 5cm from wall")
        print("      • 10cm from wall")
        print("      • 15cm from wall")
        print("      • 20cm from wall")
        print("      • 25cm from wall") 
        print("      • 30cm from wall")
        print("   3. Keep robot perfectly aligned (perpendicular to wall)")
        print("   4. Avoid shadows on IR sensors")
        print("   5. Press Enter when robot is positioned correctly at each distance")
        print("")
        
        platform_data = {}
        
        for distance_cm in self.distances_cm:
            print(f"\n🎯 Distance: {distance_cm}cm")
            print(f"   📏 Position robot {distance_cm}cm from wall using your ruler or measuring tape")
            print(f"   📐 Ensure robot is perpendicular to wall surface")
            print(f"   💡 Check lighting - avoid shadows on sensors")
            input(f"   Press Enter when robot is positioned at {distance_cm}cm: ")
            
            # Collect readings
            sensor_data = self.collect_sensor_readings(robot, distance_cm, self.readings_per_distance)
            platform_data[distance_cm] = sensor_data
            
            # Show primary sensor readings
            front_sensors_mean = []
            for sensor_idx in self.primary_sensors:
                if sensor_data[sensor_idx]["mean"] is not None:
                    front_sensors_mean.append(sensor_data[sensor_idx]["mean"])
            
            if front_sensors_mean:
                avg_front = np.mean(front_sensors_mean)
                print(f"   ✅ Average front sensor reading: {avg_front:.2f}")
            else:
                print("   ⚠️  No front sensor readings detected")
        
        self.calibration_data["hardware"] = platform_data
        print("\n✅ Hardware calibration complete!")
    
    def analyze_calibration_data(self):
        """Analyze collected calibration data and generate recommendations"""
        print("\n📊 CALIBRATION ANALYSIS")
        print("=" * 50)
        
        # Check if we have data for both platforms
        sim_data = self.calibration_data.get("simulation", {})
        hw_data = self.calibration_data.get("hardware", {})
        
        if not sim_data:
            print("❌ No simulation data found")
            return
        
        if not hw_data:
            print("❌ No hardware data found")
            return
        
        print(f"📈 Analyzing data for {len(self.distances_cm)} distance points...")
        
        # Analysis for each distance
        for distance_cm in self.distances_cm:
            if distance_cm in sim_data and distance_cm in hw_data:
                print(f"\n📏 Distance: {distance_cm}cm")
                
                # Get front sensor averages
                sim_front_readings = []
                hw_front_readings = []
                
                for sensor_idx in self.primary_sensors:
                    if (sim_data[distance_cm][sensor_idx]["mean"] is not None and
                        hw_data[distance_cm][sensor_idx]["mean"] is not None):
                        sim_front_readings.append(sim_data[distance_cm][sensor_idx]["mean"])
                        hw_front_readings.append(hw_data[distance_cm][sensor_idx]["mean"])
                
                if sim_front_readings and hw_front_readings:
                    sim_avg = np.mean(sim_front_readings)
                    hw_avg = np.mean(hw_front_readings)
                    
                    print(f"   🎮 Simulation avg: {sim_avg:.2f}")
                    print(f"   🤖 Hardware avg:   {hw_avg:.2f}")
                    
                    # Calculate normalized values
                    sim_normalized = sim_avg / 2000.0  # Simulation max range
                    hw_normalized = 1.0 - (hw_avg / 100.0)  # Hardware (inverted)
                    
                    print(f"   📊 Sim normalized: {sim_normalized:.3f}")
                    print(f"   📊 HW normalized:  {hw_normalized:.3f}")
                    print(f"   🔄 Difference:     {abs(sim_normalized - hw_normalized):.3f}")
        
        # Generate threshold recommendations
        self.generate_threshold_recommendations()
    
    def generate_threshold_recommendations(self):
        """Generate threshold recommendations for different target distances"""
        print("\n🎯 THRESHOLD RECOMMENDATIONS")
        print("=" * 50)
        
        target_distances = [10, 12, 15]  # Common safety distances
        
        sim_data = self.calibration_data.get("simulation", {})
        hw_data = self.calibration_data.get("hardware", {})
        
        for target_dist in target_distances:
            print(f"\n🛡️  Target Safety Distance: {target_dist}cm")
            
            # Find closest calibrated distance
            closest_dist = min(self.distances_cm, key=lambda x: abs(x - target_dist))
            
            if closest_dist in sim_data and closest_dist in hw_data:
                # Calculate recommended thresholds
                sim_front_readings = []
                hw_front_readings = []
                
                for sensor_idx in self.primary_sensors:
                    if (sim_data[closest_dist][sensor_idx]["mean"] is not None and
                        hw_data[closest_dist][sensor_idx]["mean"] is not None):
                        sim_front_readings.append(sim_data[closest_dist][sensor_idx]["mean"])
                        hw_front_readings.append(hw_data[closest_dist][sensor_idx]["mean"])
                
                if sim_front_readings and hw_front_readings:
                    sim_threshold = np.mean(sim_front_readings)
                    hw_threshold = np.mean(hw_front_readings)
                    
                    print(f"   📏 Using calibration from {closest_dist}cm")
                    print(f"   🎮 Simulation threshold: {sim_threshold:.0f}")
                    print(f"   🤖 Hardware threshold:   {hw_threshold:.0f}")
                    
                    # Show normalized comparison
                    sim_norm = sim_threshold / 2000.0
                    hw_norm = 1.0 - (hw_threshold / 100.0)
                    print(f"   📊 Normalized difference: {abs(sim_norm - hw_norm):.3f}")
                    
                    # Code snippet
                    print(f"\n   💻 Code for {target_dist}cm safety:")
                    print(f"   if self.is_simulation:")
                    print(f"       self.obstacle_threshold = {sim_threshold:.0f}")
                    print(f"   else:")
                    print(f"       self.obstacle_threshold = {hw_threshold:.0f}")
    
    def save_calibration_data(self):
        """Save calibration data to JSON file"""
        print(f"\n💾 Saving calibration data to {self.calibration_file}")
        
        with open(self.calibration_file, 'w') as f:
            json.dump(self.calibration_data, f, indent=2, default=str)
        
        print(f"✅ Calibration data saved successfully!")
        print(f"📁 File location: {self.calibration_file.absolute()}")
    
    def load_calibration_data(self) -> bool:
        """Load existing calibration data from JSON file"""
        if self.calibration_file.exists():
            print(f"📂 Loading existing calibration data from {self.calibration_file}")
            
            with open(self.calibration_file, 'r') as f:
                self.calibration_data = json.load(f)
            
            print("✅ Calibration data loaded successfully!")
            return True
        else:
            print(f"📄 No existing calibration file found")
            return False

    def get_threshold_for_distance(self, target_distance_cm: float, platform: str) -> Optional[float]:
        """Get threshold value that corresponds to a specific physical distance for a platform"""
        platform_data = self.calibration_data.get(platform, {})
        
        if not platform_data:
            return None
        
        # Find closest calibrated distance
        closest_dist = min(self.distances_cm, key=lambda x: abs(x - target_distance_cm))
        
        if closest_dist not in platform_data:
            return None
        
        # Calculate average front sensor reading for this distance
        front_readings = []
        for sensor_idx in self.primary_sensors:
            if platform_data[closest_dist][sensor_idx]["mean"] is not None:
                front_readings.append(platform_data[closest_dist][sensor_idx]["mean"])
        
        if front_readings:
            return np.mean(front_readings)
        
        return None
    
    def validate_threshold_consistency(self, target_distance_cm: float) -> Dict:
        """Validate that thresholds for both platforms represent the same physical distance"""
        sim_threshold = self.get_threshold_for_distance(target_distance_cm, "simulation")
        hw_threshold = self.get_threshold_for_distance(target_distance_cm, "hardware")
        
        if sim_threshold is None or hw_threshold is None:
            return {
                "valid": False,
                "reason": "Missing calibration data for one or both platforms",
                "sim_threshold": sim_threshold,
                "hw_threshold": hw_threshold
            }
        
        # Calculate normalized values (0-1 scale where 0=obstacle, 1=clear)
        sim_normalized = sim_threshold / 2000.0  # Simulation: higher values = farther
        hw_normalized = 1.0 - (hw_threshold / 100.0)  # Hardware: lower values = farther (inverted)
        
        # Check if normalized values are close (within 5% tolerance)
        difference = abs(sim_normalized - hw_normalized)
        tolerance = 0.05  # 5% tolerance
        
        return {
            "valid": difference <= tolerance,
            "difference": difference,
            "tolerance": tolerance,
            "sim_threshold": sim_threshold,
            "hw_threshold": hw_threshold,
            "sim_normalized": sim_normalized,
            "hw_normalized": hw_normalized,
            "target_distance_cm": target_distance_cm
        }
    
    def get_initial_thresholds(self, target_distance_cm: float = 12.0) -> Dict:
        """Get initial threshold recommendations before full calibration"""
        print(f"\n🎯 INITIAL THRESHOLD ESTIMATION for {target_distance_cm}cm safety distance")
        print("=" * 60)
        
        # Default fallback thresholds based on typical sensor characteristics
        fallback_thresholds = {
            "simulation": {
                5: 1800,     # Very close
                10: 1600,    # Close  
                15: 1400,    # Medium
                20: 1200,    # Medium-far
                25: 1000,    # Far
                30: 800      # Maximum range
            },
            "hardware": {
                5: 80,       # Very close
                10: 60,      # Close
                15: 45,      # Medium  
                20: 30,      # Medium-far
                25: 15,      # Far
                30: 8        # Maximum range
            }
        }
        
        # Find closest distance in our fallback table
        closest_dist = min(fallback_thresholds["simulation"].keys(), 
                          key=lambda x: abs(x - target_distance_cm))
        
        sim_threshold = fallback_thresholds["simulation"][closest_dist]
        hw_threshold = fallback_thresholds["hardware"][closest_dist]
        
        # Calculate normalized values
        sim_normalized = sim_threshold / 2000.0
        hw_normalized = 1.0 - (hw_threshold / 100.0)
        
        print(f"📏 Estimated thresholds for {target_distance_cm}cm (using {closest_dist}cm reference):")
        print(f"   🎮 Simulation: {sim_threshold} (normalized: {sim_normalized:.3f})")
        print(f"   🤖 Hardware:   {hw_threshold} (normalized: {hw_normalized:.3f})")
        print(f"   🔄 Difference: {abs(sim_normalized - hw_normalized):.3f}")
        print(f"\n💡 These are ESTIMATES - run full calibration for accurate values!")
        
        return {
            "sim_threshold": sim_threshold,
            "hw_threshold": hw_threshold,
            "sim_normalized": sim_normalized,
            "hw_normalized": hw_normalized,
            "target_distance_cm": target_distance_cm,
            "source": "estimated_fallback"
        }
    
    def generate_calibrated_thresholds(self) -> Dict:
        """Generate final calibrated thresholds for different safety distances"""
        print("\n🎯 CALIBRATED THRESHOLDS FOR REALITY GAP REDUCTION")
        print("=" * 60)
        
        target_distances = [5, 10, 12.0, 15, 20]  # Include standard distances plus 12cm safety
        calibrated_thresholds = {}
        
        for target_dist in target_distances:
            validation = self.validate_threshold_consistency(target_dist)
            
            if validation["valid"]:
                print(f"\n✅ {target_dist}cm Safety Distance - GOOD ALIGNMENT")
                print(f"   🎮 Simulation threshold: {validation['sim_threshold']:.0f}")
                print(f"   🤖 Hardware threshold:   {validation['hw_threshold']:.0f}")
                print(f"   📊 Normalized difference: {validation['difference']:.3f} (< {validation['tolerance']:.2f})")
                
                calibrated_thresholds[target_dist] = {
                    "sim_threshold": int(validation['sim_threshold']),
                    "hw_threshold": int(validation['hw_threshold']),
                    "quality": "excellent"
                }
            else:
                print(f"\n⚠️  {target_dist}cm Safety Distance - NEEDS ADJUSTMENT")
                if validation.get("sim_threshold") and validation.get("hw_threshold"):
                    print(f"   🎮 Simulation threshold: {validation['sim_threshold']:.0f}")
                    print(f"   🤖 Hardware threshold:   {validation['hw_threshold']:.0f}")
                    print(f"   📊 Normalized difference: {validation['difference']:.3f} (> {validation['tolerance']:.2f})")
                    
                    calibrated_thresholds[target_dist] = {
                        "sim_threshold": int(validation['sim_threshold']),
                        "hw_threshold": int(validation['hw_threshold']),
                        "quality": "needs_adjustment"
                    }
                else:
                    print(f"   ❌ Missing calibration data")
        
        # Generate ready-to-use code
        if calibrated_thresholds:
            print(f"\n💻 READY-TO-USE CODE (copy to your RobotEnvironment.__init__):")
            print("=" * 60)
            print("# CALIBRATED THRESHOLDS - Reduces reality gap by ensuring")
            print("# both platforms use thresholds representing same physical distance")
            print("")
            
            # Find best threshold (prefer 12cm, fallback to others)
            best_distance = 12.0 if 12.0 in calibrated_thresholds else list(calibrated_thresholds.keys())[0]
            best_thresholds = calibrated_thresholds[best_distance]
            
            print(f"# Calibrated for {best_distance}cm safety distance")
            print("if self.is_simulation:")
            print(f"    self.obstacle_threshold = {best_thresholds['sim_threshold']}")
            print("else:")
            print(f"    self.obstacle_threshold = {best_thresholds['hw_threshold']}")
            print("")
            print(f"# Quality: {best_thresholds['quality']}")
        
        return calibrated_thresholds

def main():
    """Main calibration interface"""
    print("🔧 IR SENSOR CALIBRATION TOOL")
    print("=" * 50)
    print("This tool helps calibrate IR sensor thresholds between simulation and hardware")
    print("for consistent obstacle detection across platforms.")
    print("")
    print("🔧 SETUP REQUIREMENTS:")
    print("  For Simulation:")
    print("    - CoppeliaSim running with robot scene loaded")
    print("    - API server accessible (default port 23000)")
    print("  For Hardware:")
    print("    - Robot connected and powered on")
    print("    - ROS system running")
    print("    - Robot services accessible")
    print("")
    
    calibrator = IRSensorCalibrator()
    
    # Load existing data if available
    has_existing_data = calibrator.load_calibration_data()
    
    if has_existing_data:
        print("\n📊 Existing calibration data found!")
        choice = input("Do you want to (1) Add new data, (2) Analyze existing data, or (3) Start fresh? [1/2/3]: ")
        if choice == "3":
            calibrator.calibration_data = {
                "simulation": {},
                "hardware": {},
                "metadata": calibrator.calibration_data["metadata"]
            }
        elif choice == "2":
            calibrator.analyze_calibration_data()
            return
    
    print("\n🤖 Robot Platform Selection:")
    print("1. Simulation Robot (CoppeliaSim)")
    print("2. Hardware Robot (Physical)")
    print("3. Analyze existing calibration data")
    print("4. Get initial threshold estimates")
    print("5. Exit")
    
    while True:
        choice = input("\nSelect option [1-5]: ").strip()
        
        if choice == "1":
            print("\n🎮 Initializing Simulation Robot...")
            try:
                robot = SimulationRobobo()
                calibrator.calibrate_simulation(robot)
                break
            except Exception as e:
                print(f"❌ Error connecting to simulation: {e}")
                print("   Make sure:")
                print("   - CoppeliaSim is running")
                print("   - Robot scene is loaded")
                print("   - API server is accessible (default port 23000)")
                continue
        
        elif choice == "2":
            print("\n🤖 Initializing Hardware Robot...")
            try:
                # Use camera=False for calibration since we only need IR sensors
                robot = HardwareRobobo(camera=False)
                calibrator.calibrate_hardware(robot)
                break
            except Exception as e:
                print(f"❌ Error connecting to hardware: {e}")
                print("   Make sure robot is connected and powered on")
                print("   Check ROS connection and robot services are running")
                continue
        
        elif choice == "3":
            if has_existing_data:
                calibrator.analyze_calibration_data()
                # Also generate final calibrated thresholds
                calibrator.generate_calibrated_thresholds()
            else:
                print("❌ No existing calibration data found")
            return
        
        elif choice == "4":
            # Get initial threshold estimates
            target_distance = input("Enter target safety distance in cm (default 12): ").strip()
            try:
                target_distance = float(target_distance) if target_distance else 12.0
            except ValueError:
                target_distance = 12.0
            
            calibrator.get_initial_thresholds(target_distance)
            return
        
        elif choice == "5":
            print("👋 Exiting calibration tool")
            return
        
        else:
            print("❌ Invalid choice. Please select 1-5.")
    
    # Save the collected data
    calibrator.save_calibration_data()
    
    # Run analysis if we have data for both platforms
    sim_data = calibrator.calibration_data.get("simulation", {})
    hw_data = calibrator.calibration_data.get("hardware", {})
    
    if sim_data and hw_data:
        print("\n🎉 Both platforms calibrated! Running analysis...")
        calibrator.analyze_calibration_data()
        # Generate final calibrated thresholds for reality gap reduction
        calibrator.generate_calibrated_thresholds()
    elif sim_data or hw_data:
        print("\n📝 Partial calibration complete.")
        print("   Run the tool again to calibrate the other platform for full analysis.")
    
    print(f"\n📁 Calibration data saved to: {calibrator.calibration_file.absolute()}")
    print("💡 Use this data to update your robot environment thresholds!")


if __name__ == "__main__":
    main()
