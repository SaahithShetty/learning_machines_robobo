#!/usr/bin/env bash
source /opt/ros/noetic/setup.bash
source /root/catkin_ws/devel/setup.bash
source /root/catkin_ws/setup.bash

# Check if we have a direct python script execution command
if [[ "$#" -eq 1 ]] && [[ "$1" == *"python3"* ]] && [[ "$1" == *".py"* ]]; then
    # Parse the full command string and execute it
    eval "$1"
elif [[ "$1" == "python3" ]] && [[ "$2" == *"/train_rl.py" ]]; then
    # Direct python script execution for training
    exec "$@"
elif [[ "$1" == *"/train_rl.py" ]]; then
    # Python script execution (add python3 prefix)
    exec python3 "$@"
elif [[ "$1" == *".py" ]]; then
    # Other python script execution
    exec python3 "$@"
else
    # Default: use the learning_robobo_controller
    exec rosrun learning_machines learning_robobo_controller.py "$@"
fi
