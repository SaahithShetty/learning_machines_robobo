#!/usr/bin/env python3
"""
Simple test for the Emotion enum fix
"""

import sys
from pathlib import Path

# Add the robobo_interface module to path
sys.path.append('/Users/saahithshetty/Documents/Coding/Learning_Machines/learning_machines_robobo/examples/full_project_setup/catkin_ws/src/robobo_interface/src')

try:
    from robobo_interface.datatypes import Emotion
    
    print("🤖 EMOTION SYSTEM TEST")
    print("="*50)
    
    print("Testing fixed Emotion enum:")
    emotions = [
        Emotion.HAPPY,
        Emotion.LAUGHING,  # This should work now (was LAUCHING)
        Emotion.SAD,
        Emotion.ANGRY,
        Emotion.SURPRISED,
        Emotion.NORMAL
    ]
    
    for emotion in emotions:
        print(f"  ✅ {emotion.name}: '{emotion.value}'")
    
    print("\n✅ SUCCESS: All emotions are correctly defined!")
    print("The typo 'LAUCHING' has been fixed to 'LAUGHING'")
    
except Exception as e:
    print(f"❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
