#!/usr/bin/env python3
"""
CoppeliaSim Scene Builder for IR Sensor Calibration Arena
=========================================================

This script programmatically creates a calibration arena scene for Robobo IR sensor testing.
It creates a clean environment with:
- Flat floor plane
- Calibration wall for distance measurements
- Colored distance markers at 5cm intervals (5, 10, 15, 20, 25, 30cm)
- Proper lighting and camera positions
- Robot starting position

Usage:
1. Start CoppeliaSim
2. Run this script
3. Scene will be created automatically
4. Save as 'arena_calibration.ttt'
"""

import sys
import time
import math

# Add CoppeliaSim remote API path
sys.path.append('/opt/CoppeliaSim_Edu_V4_1_0_Ubuntu18_04/programming/remoteApiBindings/python/python')

try:
    import sim
except ImportError:
    print("ERROR: Could not import CoppeliaSim remote API")
    print("Make sure CoppeliaSim is installed and PYTHONPATH is set correctly")
    sys.exit(1)


class CalibrationArenaBuilder:
    """Builder class for creating calibration arena in CoppeliaSim"""
    
    def __init__(self):
        self.client_id = -1
        self.objects = {}
        
    def connect(self):
        """Connect to CoppeliaSim"""
        print("Connecting to CoppeliaSim...")
        
        # Close any existing connections
        sim.simxFinish(-1)
        
        # Connect to CoppeliaSim
        self.client_id = sim.simxStart('127.0.0.1', 19997, True, True, 5000, 5)
        
        if self.client_id != -1:
            print("✓ Connected to CoppeliaSim successfully")
            return True
        else:
            print("✗ Failed to connect to CoppeliaSim")
            return False
    
    def disconnect(self):
        """Disconnect from CoppeliaSim"""
        if self.client_id != -1:
            sim.simxFinish(self.client_id)
            print("Disconnected from CoppeliaSim")
    
    def clear_scene(self):
        """Clear the current scene"""
        print("Clearing scene...")
        sim.simxStopSimulation(self.client_id, sim.simx_opmode_blocking)
        time.sleep(0.5)
        
        # Get all objects in scene
        ret_code, handles = sim.simxGetObjectsInTree(self.client_id, sim.sim_handle_scene, sim.sim_handle_all, 0, sim.simx_opmode_blocking)
        
        if ret_code == sim.simx_return_ok:
            for handle in handles:
                # Don't delete built-in objects like default cameras/lights
                ret_code, name = sim.simxGetObjectName(self.client_id, handle, sim.simx_opmode_blocking)
                if ret_code == sim.simx_return_ok and not name.startswith('DefaultCamera') and not name.startswith('DefaultLight'):
                    sim.simxRemoveObject(self.client_id, handle, sim.simx_opmode_blocking)
        
        print("✓ Scene cleared")
    
    def create_floor(self):
        """Create floor plane"""
        print("Creating floor...")
        
        # Create plane primitive
        ret_code, floor_handle = sim.simxCreatePrimitiveShape(
            self.client_id,
            sim.sim_primitiveshape_plane,
            [5.0, 5.0, 0.1],  # 5m x 5m x 0.1m
            1.0,  # mass
            sim.simx_opmode_blocking
        )
        
        if ret_code == sim.simx_return_ok:
            # Set floor position
            sim.simxSetObjectPosition(self.client_id, floor_handle, -1, [0, 0, 0], sim.simx_opmode_blocking)
            
            # Set floor color (light gray)
            sim.simxSetShapeColor(self.client_id, floor_handle, None, sim.sim_colorcomponent_ambient_diffuse, [0.8, 0.8, 0.8], sim.simx_opmode_blocking)
            
            # Set name
            sim.simxSetObjectName(self.client_id, floor_handle, "CalibrationFloor", sim.simx_opmode_blocking)
            
            self.objects['floor'] = floor_handle
            print("✓ Floor created")
        else:
            print("✗ Failed to create floor")
    
    def create_calibration_wall(self):
        """Create main calibration wall"""
        print("Creating calibration wall...")
        
        # Create cuboid primitive
        ret_code, wall_handle = sim.simxCreatePrimitiveShape(
            self.client_id,
            sim.sim_primitiveshape_cuboid,
            [0.1, 3.0, 1.0],  # 0.1m thick, 3m wide, 1m tall
            0.0,  # static (no mass)
            sim.simx_opmode_blocking
        )
        
        if ret_code == sim.simx_return_ok:
            # Position wall 2m forward from center
            sim.simxSetObjectPosition(self.client_id, wall_handle, -1, [2.0, 0, 0.5], sim.simx_opmode_blocking)
            
            # Set wall color (white)
            sim.simxSetShapeColor(self.client_id, wall_handle, None, sim.sim_colorcomponent_ambient_diffuse, [0.9, 0.9, 0.9], sim.simx_opmode_blocking)
            
            # Set name
            sim.simxSetObjectName(self.client_id, wall_handle, "CalibrationWall", sim.simx_opmode_blocking)
            
            self.objects['wall'] = wall_handle
            print("✓ Calibration wall created")
        else:
            print("✗ Failed to create calibration wall")
    
    def create_distance_markers(self):
        """Create colored distance markers"""
        print("Creating distance markers...")
        
        # Marker specifications: [distance_cm, color_rgb, name]
        markers_spec = [
            (5, [1.0, 0.0, 0.0], "Marker_5cm"),     # Red
            (10, [0.0, 1.0, 0.0], "Marker_10cm"),   # Green  
            (15, [0.0, 0.0, 1.0], "Marker_15cm"),   # Blue
            (20, [1.0, 1.0, 0.0], "Marker_20cm"),   # Yellow
            (25, [1.0, 0.0, 1.0], "Marker_25cm"),   # Magenta
            (30, [1.0, 0.5, 0.0], "Marker_30cm"),   # Orange
        ]
        
        wall_x = 2.0  # Wall position
        
        for distance_cm, color, name in markers_spec:
            distance_m = distance_cm / 100.0
            marker_x = wall_x - 0.05 - distance_m  # Wall surface - wall_thickness/2 - distance
            
            # Create cylinder marker
            ret_code, marker_handle = sim.simxCreatePrimitiveShape(
                self.client_id,
                sim.sim_primitiveshape_cylinder,
                [0.02, 0.02, 0.1],  # 2cm diameter, 10cm height
                0.0,  # static
                sim.simx_opmode_blocking
            )
            
            if ret_code == sim.simx_return_ok:
                # Position marker
                sim.simxSetObjectPosition(self.client_id, marker_handle, -1, [marker_x, 0, 0.05], sim.simx_opmode_blocking)
                
                # Set color
                sim.simxSetShapeColor(self.client_id, marker_handle, None, sim.sim_colorcomponent_ambient_diffuse, color, sim.simx_opmode_blocking)
                
                # Set name
                sim.simxSetObjectName(self.client_id, marker_handle, name, sim.simx_opmode_blocking)
                
                self.objects[f'marker_{distance_cm}cm'] = marker_handle
                print(f"✓ Created {name} at {distance_cm}cm")
            else:
                print(f"✗ Failed to create {name}")
    
    def create_robot_start_position(self):
        """Create dummy object marking robot start position"""
        print("Creating robot start position marker...")
        
        # Create small sphere as position marker
        ret_code, start_handle = sim.simxCreatePrimitiveShape(
            self.client_id,
            sim.sim_primitiveshape_sphere,
            [0.05, 0.05, 0.05],  # 5cm sphere
            0.0,  # static
            sim.simx_opmode_blocking
        )
        
        if ret_code == sim.simx_return_ok:
            # Position 40cm from wall (at 1.20m from origin)
            sim.simxSetObjectPosition(self.client_id, start_handle, -1, [1.20, 0, 0.025], sim.simx_opmode_blocking)
            
            # Set color (gray)
            sim.simxSetShapeColor(self.client_id, start_handle, None, sim.sim_colorcomponent_ambient_diffuse, [0.5, 0.5, 0.5], sim.simx_opmode_blocking)
            
            # Set name
            sim.simxSetObjectName(self.client_id, start_handle, "RoboboStartPosition", sim.simx_opmode_blocking)
            
            self.objects['start_position'] = start_handle
            print("✓ Robot start position marked")
        else:
            print("✗ Failed to create start position marker")
    
    def setup_lighting(self):
        """Configure scene lighting"""
        print("Setting up lighting...")
        
        # Create directional light
        ret_code, light_handle = sim.simxCreateDummy(self.client_id, 0.1, None, sim.simx_opmode_blocking)
        
        if ret_code == sim.simx_return_ok:
            # Position light above scene
            sim.simxSetObjectPosition(self.client_id, light_handle, -1, [1.5, 0, 3.0], sim.simx_opmode_blocking)
            
            # Set name
            sim.simxSetObjectName(self.client_id, light_handle, "CalibrationLight", sim.simx_opmode_blocking)
            
            self.objects['light'] = light_handle
            print("✓ Lighting configured")
        else:
            print("✗ Failed to create lighting")
    
    def setup_cameras(self):
        """Setup calibration camera views"""
        print("Setting up cameras...")
        
        # Overview camera
        ret_code, overview_cam = sim.simxCreateDummy(self.client_id, 0.1, None, sim.simx_opmode_blocking)
        if ret_code == sim.simx_return_ok:
            sim.simxSetObjectPosition(self.client_id, overview_cam, -1, [0, -3, 2], sim.simx_opmode_blocking)
            sim.simxSetObjectName(self.client_id, overview_cam, "OverviewCamera", sim.simx_opmode_blocking)
            self.objects['overview_cam'] = overview_cam
        
        # Side camera  
        ret_code, side_cam = sim.simxCreateDummy(self.client_id, 0.1, None, sim.simx_opmode_blocking)
        if ret_code == sim.simx_return_ok:
            sim.simxSetObjectPosition(self.client_id, side_cam, -1, [3, 1, 1], sim.simx_opmode_blocking)
            sim.simxSetObjectName(self.client_id, side_cam, "SideCamera", sim.simx_opmode_blocking)
            self.objects['side_cam'] = side_cam
        
        print("✓ Camera positions marked")
    
    def build_arena(self):
        """Build complete calibration arena"""
        print("\n🔧 Building Calibration Arena for Robobo IR Sensor Testing")
        print("=" * 60)
        
        if not self.connect():
            return False
        
        try:
            # Build scene components
            self.clear_scene()
            self.create_floor()
            self.create_calibration_wall() 
            self.create_distance_markers()
            self.create_robot_start_position()
            self.setup_lighting()
            self.setup_cameras()
            
            print("\n" + "=" * 60)
            print("🎉 Calibration Arena Created Successfully!")
            print("\nNext Steps:")
            print("1. Save scene as 'arena_calibration.ttt'")
            print("2. Add Robobo robot model to the scene")
            print("3. Position robot at gray start marker")
            print("4. Run calibration script")
            print("\nDistance Markers:")
            print("  🔴 Red = 5cm    🟢 Green = 10cm   🔵 Blue = 15cm")
            print("  🟡 Yellow = 20cm   🟣 Magenta = 25cm   🟠 Orange = 30cm")
            
            return True
            
        except Exception as e:
            print(f"\n✗ Error building arena: {e}")
            return False
        
        finally:
            # Keep connection open for user to save scene
            print(f"\nConnection kept open. Save the scene and close CoppeliaSim when ready.")
            print("Press Enter to disconnect...")
            input()
            self.disconnect()


def main():
    """Main function"""
    builder = CalibrationArenaBuilder()
    
    print("🤖 Robobo IR Sensor Calibration Arena Builder")
    print("This will create a new calibration scene in CoppeliaSim")
    print("\nPrerequisites:")
    print("✓ CoppeliaSim must be running")
    print("✓ No other simulation should be active")
    print("✓ Remote API should be enabled")
    
    input("\nPress Enter to continue...")
    
    success = builder.build_arena()
    
    if success:
        print("\n✓ Arena creation completed successfully!")
    else:
        print("\n✗ Arena creation failed. Check CoppeliaSim connection.")
    

if __name__ == "__main__":
    main()
