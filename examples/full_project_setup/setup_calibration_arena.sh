#!/bin/bash

# Quick Calibration Arena Setup Script
# ===================================
# This script helps prepare the environment for IR sensor calibration

echo "🤖 Robobo IR Sensor Calibration Arena Setup"
echo "==========================================="
echo

# Check if we're in Docker container
if [ -f /.dockerenv ]; then
    echo "✓ Running inside Docker container"
    RESULTS_DIR="/root/results"
else
    echo "ℹ Running on host system"
    RESULTS_DIR="./results"
fi

# Create results directory if it doesn't exist
if [ ! -d "$RESULTS_DIR" ]; then
    mkdir -p "$RESULTS_DIR"
    echo "✓ Created results directory: $RESULTS_DIR"
else
    echo "✓ Results directory exists: $RESULTS_DIR"
fi

# Check for existing calibration data
CALIB_FILE="$RESULTS_DIR/ir_sensor_calibration_data.json"
if [ -f "$CALIB_FILE" ]; then
    echo "⚠ Existing calibration data found: $CALIB_FILE"
    echo "  File will be backed up before new calibration"
    BACKUP_FILE="$RESULTS_DIR/ir_sensor_calibration_data_backup_$(date +%Y%m%d_%H%M%S).json"
    cp "$CALIB_FILE" "$BACKUP_FILE"
    echo "  Backup created: $BACKUP_FILE"
fi

echo
echo "📋 Arena Creation Options:"
echo "1. Manual creation using CoppeliaSim GUI (Recommended)"
echo "2. Programmatic creation using Python script"
echo "3. Import XML scene file"

echo
echo "🎯 For Manual Creation:"
echo "- Follow guide: scenes/manual_arena_creation_guide.md"
echo "- Use existing arena_obstacles.ttt as base"
echo "- Create calibration wall and distance markers"
echo "- Position robot at start marker"

echo
echo "🔧 For Programmatic Creation:"
echo "- Ensure CoppeliaSim is running"
echo "- Run: python create_calibration_arena.py"
echo "- Save the generated scene as arena_calibration.ttt"

echo
echo "📥 For XML Import:"
echo "- Use provided arena_calibration.xml file"
echo "- File → Import → Model/Scene in CoppeliaSim"
echo "- Add Robobo robot model manually"

echo
echo "📐 Distance Markers Reference:"
echo "🔴 Red    = 5cm  (X: 1.85)"
echo "🟢 Green  = 10cm (X: 1.80)" 
echo "🔵 Blue   = 15cm (X: 1.75)"
echo "🟡 Yellow = 20cm (X: 1.70)"
echo "🟣 Magenta= 25cm (X: 1.65)"
echo "🟠 Orange = 30cm (X: 1.60)"
echo "⚪ Start  = 40cm (X: 1.20)"

echo
echo "🚀 Next Steps:"
echo "1. Create calibration arena using preferred method"
echo "2. Load arena_calibration.ttt in CoppeliaSim"  
echo "3. Position Robobo robot at gray start marker"
echo "4. Ensure robot faces the calibration wall"
echo "5. Run calibration: python sensor_calibration_tool.py"

echo
echo "🔍 Troubleshooting:"
echo "- If sensors read 0.0: Check robot connection and scene setup"
echo "- Use calibration_diagnostic.py to test sensor connectivity"
echo "- Verify IR sensors are enabled in robot model"
echo "- Ensure proper lighting and wall material properties"

echo
echo "📁 Generated Files:"
echo "- Calibration data: $CALIB_FILE"
echo "- Diagnostic logs: $RESULTS_DIR/diagnostic_*.log"
echo "- Backup files: $RESULTS_DIR/*_backup_*.json"

echo
echo "Ready for calibration arena setup!"
echo "Press Enter to continue or Ctrl+C to exit"
read -r

# Show current directory structure for reference
echo
echo "📂 Current workspace structure:"
find . -name "*.py" -o -name "*.ttt" -o -name "*.md" -o -name "*.json" | grep -E "(sensor|calibr|arena)" | head -20

echo
echo "Setup complete! Proceed with arena creation."
