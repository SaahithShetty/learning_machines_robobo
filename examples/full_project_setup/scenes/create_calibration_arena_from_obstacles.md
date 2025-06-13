# Creating Calibration Arena from arena_obstacles.ttt

## Overview
This guide shows how to modify the existing `arena_obstacles.ttt` scene to create a proper calibration arena for IR sensor testing.

## Step-by-Step Instructions

### 1. Open and Prepare Scene
1. **Open CoppeliaSim**
2. **Load** `arena_obstacles.ttt` from your scenes folder
3. **Save As** → `arena_calibration.ttt` (keep original intact)

### 2. Clear Existing Obstacles
1. **Select all obstacle objects** in the scene hierarchy
2. **Delete** all random obstacles/walls except:
   - Floor/ground plane
   - Arena boundaries (if any)
   - Robobo robot
   - Lighting objects

### 3. Add Calibration Wall
1. **Add** → Primitive shape → Cuboid
2. **Size**: 0.1m (width) × 3.0m (length) × 1.0m (height)
3. **Position**: Place at one edge of the arena
4. **Material**: Set to matte, non-reflective surface
5. **Color**: Light gray or white
6. **Name**: "CalibrationWall"

### 4. Add Distance Markers
Create small cylinder markers at precise distances from the wall:

#### Red Marker (5cm):
- **Add** → Primitive shape → Cylinder
- **Size**: 0.02m diameter × 0.1m height
- **Position**: 5cm from wall surface
- **Color**: Red (RGB: 1.0, 0.0, 0.0)
- **Name**: "Marker_5cm"

#### Green Marker (10cm):
- **Size**: 0.02m diameter × 0.1m height
- **Position**: 10cm from wall surface
- **Color**: Green (RGB: 0.0, 1.0, 0.0)
- **Name**: "Marker_10cm"

#### Blue Marker (15cm):
- **Size**: 0.02m diameter × 0.1m height
- **Position**: 15cm from wall surface
- **Color**: Blue (RGB: 0.0, 0.0, 1.0)
- **Name**: "Marker_15cm"

#### Yellow Marker (20cm):
- **Size**: 0.02m diameter × 0.1m height
- **Position**: 20cm from wall surface
- **Color**: Yellow (RGB: 1.0, 1.0, 0.0)
- **Name**: "Marker_20cm"

#### Purple Marker (25cm):
- **Size**: 0.02m diameter × 0.1m height
- **Position**: 25cm from wall surface
- **Color**: Purple (RGB: 1.0, 0.0, 1.0)
- **Name**: "Marker_25cm"

#### Orange Marker (30cm):
- **Size**: 0.02m diameter × 0.1m height
- **Position**: 30cm from wall surface
- **Color**: Orange (RGB: 1.0, 0.5, 0.0)
- **Name**: "Marker_30cm"

### 5. Position Robot
1. **Select** the Robobo robot
2. **Position**: ~40cm from the calibration wall
3. **Orientation**: Facing the wall perpendicularly
4. **Height**: On the floor surface (Z = robot_height/2)

### 6. Adjust Lighting
1. **Ensure uniform ambient lighting** (0.6-0.8 intensity)
2. **Minimize shadows** on the calibration wall
3. **Avoid direct harsh lighting** that could interfere with IR sensors

### 7. Scene Verification
1. **Check robot orientation** - front sensors should face the wall
2. **Verify marker positions** - use CoppeliaSim measurement tools
3. **Test simulation** - ensure robot and sensors are functional
4. **Save scene** as `arena_calibration.ttt`

## Precise Positioning Coordinates

Assuming wall is at X = 2.0m position:

| Distance | Marker X Position | Robot X Position |
|----------|------------------|------------------|
| 5cm      | 1.85m           | 1.80m           |
| 10cm     | 1.80m           | 1.75m           |
| 15cm     | 1.75m           | 1.70m           |
| 20cm     | 1.70m           | 1.65m           |
| 25cm     | 1.65m           | 1.60m           |
| 30cm     | 1.60m           | 1.55m           |

*Note: Adjust these coordinates based on your actual wall position*

## Testing the Calibration Arena

### Quick Test:
1. **Start simulation**
2. **Position robot** at 20cm marker
3. **Run diagnostic command**:
   ```bash
   zsh ./scripts/run_apple_sillicon.zsh run python3 /root/catkin_ws/src/learning_machines/src/learning_machines/calibration_diagnostic.py
   ```
4. **Verify IR readings** are non-zero and reasonable

### Expected Results:
- **IR sensor readings** should be > 0
- **Readings should decrease** as distance increases
- **Front center sensor (4)** should give consistent readings

## Troubleshooting

### If sensors still read 0.0:
1. **Check robot model** - ensure IR sensors are properly configured
2. **Verify wall material** - should be detectable by IR
3. **Check sensor range** - wall must be within detection distance
4. **Inspect Lua scripts** - IR sensor scripts might need adjustment

### If readings are inconsistent:
1. **Improve wall surface** - make it more uniform
2. **Adjust lighting** - remove harsh shadows
3. **Check robot alignment** - ensure perpendicular to wall
4. **Verify marker positions** - use precise measurements

## Using the Calibration Arena

Once created, use this arena with:

```bash
# Run full calibration
zsh ./scripts/run_apple_sillicon.zsh run python3 /root/catkin_ws/src/learning_machines/src/learning_machines/sensor_calibration_tool.py

# Select option 1 (Simulation Robot)
# Position robot at each colored marker when prompted
# Take readings at each distance
```

The calibration tool will now collect proper sensor data for threshold determination.
