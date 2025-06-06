# Task 0: Obstacle Avoidance

This task implements a simple obstacle avoidance behavior for the Robobo robot. The robot moves forward until it detects an obstacle, then turns right to avoid it.

## Running the Task

To run the task in the simulator:

```bash
# Run with the default walk_until_obstacle method
python3 task0_controller.py --simulation

# Or run with the obstacle_avoidance method
python3 task0_controller.py --simulation --method obstacle_avoidance
```

To run on the hardware robot:

```bash
# Run with the default walk_until_obstacle method
python3 task0_controller.py --hardware

# Or run with the obstacle_avoidance method
python3 task0_controller.py --hardware --method obstacle_avoidance
```

## Available Options

- `--hardware`: Run on the physical Robobo robot
- `--simulation`: Run in the CoppeliaSim simulator
- `--method`: Choose which method to use (`walk_until_obstacle` or `obstacle_avoidance`)
- `--duration`: Duration in seconds for the `walk_until_obstacle` method (default: 60)
- `--iterations`: Number of control iterations for the `obstacle_avoidance` method (default: 100)
- `--no-save`: Don't save sensor and action data (applies only to `obstacle_avoidance` method)

## Methods

1. `walk_until_obstacle`: A simple continuous approach that keeps the robot moving forward until an obstacle is detected, then turns right. This behavior continues for a specified duration.

2. `obstacle_avoidance`: A more sophisticated approach with data collection capabilities. It runs for a specific number of iterations and saves sensor and action data to a CSV file.
