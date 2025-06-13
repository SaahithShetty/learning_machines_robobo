# Calibration Arena Setup Guide

## Overview
This guide helps you create a CoppeliaSim scene optimized for IR sensor calibration using the sensor_calibration_tool.py script.

## Scene Setup Instructions

### 1. Basic Environment
- **Floor**: Add a Plane primitive, scale to 5m x 5m
- **Wall**: Add a Cuboid primitive (3m wide x 1m high x 0.1m thick)
- **Position wall** along one edge of the floor
- **Material**: Use smooth, non-reflective material for the wall

### 2. Distance Markers
Add visual markers at the following distances from the wall:
- **5cm marker**: Small red sphere/cylinder
- **10cm marker**: Small green sphere/cylinder  
- **15cm marker**: Small blue sphere/cylinder
- **20cm marker**: Small yellow sphere/cylinder
- **25cm marker**: Small purple sphere/cylinder
- **30cm marker**: Small orange sphere/cylinder

### 3. Robot Starting Position
- Place the Robobo robot facing the wall
- Position at ~40cm from wall initially
- Ensure robot is centered and perpendicular to wall

### 4. Lighting Setup
- Add uniform ambient lighting
- Avoid directional lights that create shadows on IR sensors
- Light intensity: moderate (not too bright to interfere with IR)

### 5. Scene Properties
- **Name**: arena_calibration.ttt
- **Physics**: Enable realistic physics
- **Gravity**: Standard Earth gravity (-9.81 m/s²)

## Calibration Process

### Using the Scene:
1. Start CoppeliaSim with this calibration scene
2. Run the sensor calibration tool: 
   ```bash
   zsh ./scripts/run_apple_sillicon.zsh run python3 /root/catkin_ws/src/learning_machines/src/learning_machines/sensor_calibration_tool.py
   ```
3. Select option 1 (Simulation Robot)
4. For each distance measurement:
   - Manually drag the robot to the appropriate distance marker
   - Ensure robot is perfectly perpendicular to the wall
   - Press Enter when positioned correctly
   - Tool will collect 10 readings automatically

### Distance Positioning:
- **5cm**: Robot front edge 5cm from wall (use red marker)
- **10cm**: Robot front edge 10cm from wall (use green marker)
- **15cm**: Robot front edge 15cm from wall (use blue marker)
- **20cm**: Robot front edge 20cm from wall (use yellow marker)
- **25cm**: Robot front edge 25cm from wall (use purple marker)
- **30cm**: Robot front edge 30cm from wall (use orange marker)

## Tips for Accurate Calibration

### Environment:
- Use a plain, smooth wall surface
- Avoid reflective or transparent materials
- Ensure consistent lighting
- Remove objects that might interfere with IR sensors

### Robot Positioning:
- Always ensure robot is perpendicular to wall
- Use CoppeliaSim's position/orientation tools for precision
- Double-check alignment before taking readings
- Front IR sensors should face directly toward wall

### Data Quality:
- Take readings with robot stationary
- Avoid vibrations or movement during data collection
- Ensure IR sensors are clean and unobstructed
- Verify sensor readings are non-zero and reasonable

## Expected Results

### Simulation Sensor Readings:
- **5cm**: ~1800-1900 (very close)
- **10cm**: ~1600-1700 (close)
- **15cm**: ~1400-1500 (medium)
- **20cm**: ~1200-1300 (medium-far)
- **25cm**: ~1000-1100 (far)
- **30cm**: ~800-900 (maximum range)

### Quality Indicators:
- Readings should decrease as distance increases
- Standard deviation should be low (< 50)
- All front sensors (4, 5, 7) should give similar readings
- No readings should be exactly 0.0 (indicates sensor failure)

## Troubleshooting

### If readings are all 0.0:
1. Check CoppeliaSim connection (port 23000)
2. Verify robot is properly loaded in scene
3. Ensure simulation is running
4. Check IR sensor objects exist in scene hierarchy

### If readings are inconsistent:
1. Improve lighting uniformity
2. Check wall surface material
3. Verify robot positioning accuracy
4. Ensure no obstacles between robot and wall

### If readings don't change with distance:
1. Check IR sensor range settings in CoppeliaSim
2. Verify wall is within sensor detection range
3. Check sensor calibration parameters in Lua scripts
