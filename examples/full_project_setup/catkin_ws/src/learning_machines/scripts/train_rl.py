#!/usr/bin/env python3
"""
RL Training Script Wrapper

This wrapper script calls the actual training implementation from learning_machines package.
"""

import sys
import os
import subprocess

# Forward all arguments to the actual training script
if __name__ == '__main__':
    script_path = '/root/catkin_ws/src/learning_machines/src/learning_machines/train_rl.py'
    cmd = [sys.executable, script_path] + sys.argv[1:]
    subprocess.run(cmd)
