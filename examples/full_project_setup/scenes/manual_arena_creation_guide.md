# Manual Calibration Arena Creation Guide

## Quick Start: Manual Arena Creation in CoppeliaSim

This guide provides step-by-step instructions for manually creating a calibration arena in CoppeliaSim using the GUI interface.

### Prerequisites
- CoppeliaSim Edu 4.1.0+ installed
- Existing `arena_obstacles.ttt` scene file
- Basic familiarity with CoppeliaSim interface

### Step 1: Prepare Base Scene

1. **Open CoppeliaSim**
2. **Load base scene**: `File` → `Open scene...` → Select `arena_obstacles.ttt`
3. **Save as new scene**: `File` → `Save scene as...` → Name it `arena_calibration.ttt`
4. **Clear obstacles**: Delete all existing obstacle objects, keeping only:
   - Floor/ground plane
   - Arena boundaries (if any)
   - Basic lighting

### Step 2: Create Calibration Wall

1. **Add Wall**:
   - `Add` → `Primitive shape` → `Cuboid`
   - **Properties** (right-click → `Properties`):
     - **Size**: X=0.10, Y=3.00, Z=1.00 (10cm thick, 3m wide, 1m tall)
     - **Position**: X=2.00, Y=0.00, Z=0.50 (2m forward from center)
     - **Mass**: 0 (static object)
   - **Appearance**:
     - Right-click → `Color` → Set to light gray/white (RGB: 0.9, 0.9, 0.9)
   - **Name**: Right-click → `Rename` → "CalibrationWall"

### Step 3: Create Distance Markers

Create small cylinder markers at precise distances from the wall:

#### 5cm Marker (Red)
1. `Add` → `Primitive shape` → `Cylinder`
2. **Properties**:
   - **Size**: X=0.02, Y=0.02, Z=0.10 (2cm diameter, 10cm height)
   - **Position**: X=1.85, Y=0.00, Z=0.05 (5cm from wall surface)
   - **Mass**: 0
3. **Color**: Red (RGB: 1.0, 0.0, 0.0)
4. **Name**: "Marker_5cm"

#### 10cm Marker (Green)
1. `Add` → `Primitive shape` → `Cylinder`
2. **Properties**:
   - **Size**: X=0.02, Y=0.02, Z=0.10
   - **Position**: X=1.80, Y=0.00, Z=0.05 (10cm from wall)
   - **Mass**: 0
3. **Color**: Green (RGB: 0.0, 1.0, 0.0)
4. **Name**: "Marker_10cm"

#### 15cm Marker (Blue)
1. `Add` → `Primitive shape` → `Cylinder`
2. **Properties**:
   - **Size**: X=0.02, Y=0.02, Z=0.10
   - **Position**: X=1.75, Y=0.00, Z=0.05 (15cm from wall)
   - **Mass**: 0
3. **Color**: Blue (RGB: 0.0, 0.0, 1.0)
4. **Name**: "Marker_15cm"

#### 20cm Marker (Yellow)
1. `Add` → `Primitive shape` → `Cylinder`
2. **Properties**:
   - **Size**: X=0.02, Y=0.02, Z=0.10
   - **Position**: X=1.70, Y=0.00, Z=0.05 (20cm from wall)
   - **Mass**: 0
3. **Color**: Yellow (RGB: 1.0, 1.0, 0.0)
4. **Name**: "Marker_20cm"

#### 25cm Marker (Magenta)
1. `Add` → `Primitive shape` → `Cylinder`
2. **Properties**:
   - **Size**: X=0.02, Y=0.02, Z=0.10
   - **Position**: X=1.65, Y=0.00, Z=0.05 (25cm from wall)
   - **Mass**: 0
3. **Color**: Magenta (RGB: 1.0, 0.0, 1.0)
4. **Name**: "Marker_25cm"

#### 30cm Marker (Orange)
1. `Add` → `Primitive shape` → `Cylinder`
2. **Properties**:
   - **Size**: X=0.02, Y=0.02, Z=0.10
   - **Position**: X=1.60, Y=0.00, Z=0.05 (30cm from wall)
   - **Mass**: 0
3. **Color**: Orange (RGB: 1.0, 0.5, 0.0)
4. **Name**: "Marker_30cm"

### Step 4: Add Robot Start Position Marker

1. `Add` → `Primitive shape` → `Sphere`
2. **Properties**:
   - **Size**: X=0.05, Y=0.05, Z=0.05 (5cm sphere)
   - **Position**: X=1.20, Y=0.00, Z=0.025 (40cm from wall)
   - **Mass**: 0
3. **Color**: Gray (RGB: 0.5, 0.5, 0.5)
4. **Name**: "RoboboStartPosition"

### Step 5: Add Robobo Robot

1. **Import Robobo model**:
   - `File` → `Import` → `Model...`
   - Navigate to Robobo model files (usually in `models/robots/`)
   - Select Robobo robot model
2. **Position robot**:
   - **Position**: X=1.20, Y=0.00, Z=0.05 (at start marker)
   - **Orientation**: Facing toward wall (positive X direction)

### Step 6: Configure Scene Settings

1. **Lighting**:
   - Ensure adequate ambient lighting for clear visibility
   - Add directional light if needed: `Add` → `Light` → `Directional light`

2. **Physics**:
   - `Simulation` → `Simulation settings...`
   - Ensure physics engine is enabled (Bullet recommended)
   - Set timestep to 0.005s for stable simulation

### Step 7: Save and Test

1. **Save scene**: `File` → `Save scene` (Ctrl+S)
2. **Test simulation**:
   - Start simulation: `Simulation` → `Start simulation`
   - Verify robot is responsive and sensors are active
   - Check that environment is stable

### Step 8: Verification Checklist

- [ ] Floor plane present and stable
- [ ] Calibration wall at 2.0m position
- [ ] All 6 distance markers correctly positioned and colored:
  - [ ] Red marker at 5cm (X=1.85)
  - [ ] Green marker at 10cm (X=1.80)
  - [ ] Blue marker at 15cm (X=1.75)
  - [ ] Yellow marker at 20cm (X=1.70)
  - [ ] Magenta marker at 25cm (X=1.65)
  - [ ] Orange marker at 30cm (X=1.60)
- [ ] Gray start position marker at 40cm (X=1.20)
- [ ] Robobo robot loaded and positioned at start marker
- [ ] Robot facing toward calibration wall
- [ ] Simulation runs without errors
- [ ] Scene saved as `arena_calibration.ttt`

## Usage Instructions

Once the arena is created:

1. **Position robot** at gray start marker facing the wall
2. **Run calibration script** from the Docker container:
   ```bash
   python sensor_calibration_tool.py
   ```
3. **Follow prompts** to position robot at each colored distance marker
4. **Record measurements** at each distance for threshold calculation

## Distance Reference

| Marker Color | Distance | X Position | RGB Color |
|-------------|----------|------------|-----------|
| 🔴 Red      | 5cm      | 1.85       | 1.0,0.0,0.0 |
| 🟢 Green    | 10cm     | 1.80       | 0.0,1.0,0.0 |
| 🔵 Blue     | 15cm     | 1.75       | 0.0,0.0,1.0 |
| 🟡 Yellow   | 20cm     | 1.70       | 1.0,1.0,0.0 |
| 🟣 Magenta  | 25cm     | 1.65       | 1.0,0.0,1.0 |
| 🟠 Orange   | 30cm     | 1.60       | 1.0,0.5,0.0 |

## Troubleshooting

### Common Issues:

1. **Objects appear black**: Check lighting settings and add ambient light
2. **Robot falls through floor**: Ensure floor has physics properties enabled
3. **Markers not visible**: Check size and color settings, ensure they're above floor level
4. **Simulation unstable**: Reduce timestep or check for overlapping objects
5. **Robot sensors not working**: Verify robot model is complete and sensors are enabled

### If Calibration Still Returns 0.0 Readings:

1. **Check robot model**: Ensure IR sensors are properly configured in robot model
2. **Verify sensor scripts**: Make sure sensor reading scripts are attached to robot
3. **Test robot connectivity**: Use diagnostic script to verify robot communication
4. **Check scene physics**: Ensure physics engine is running and stable
5. **Verify wall material**: Ensure wall has proper surface properties for IR reflection

This manual approach ensures you have complete control over the arena creation and can verify each step works correctly.
