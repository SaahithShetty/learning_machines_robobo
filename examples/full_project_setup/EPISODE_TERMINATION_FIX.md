# Episode Duration Fix for Robot Environment

## Issue
The robot episodes were not terminating at the intended maximum duration of 120 seconds (2 minutes). Instead, they were continuing to run for much longer (over 234 seconds in some cases).

## Identified Problems

1. Mismatch between environment and training loop timing:
   - The environment was correctly set to terminate after 120 seconds
   - The training loop had a hardcoded step limit of 1800 steps (3 minutes at 10Hz)
   - The evaluation loop had a hardcoded step limit of 3000 steps (5 minutes at 10Hz)

2. These step limits were overriding the environment's time-based termination logic.

## Fixes Implemented

1. Updated training loop step limit:
   - Changed from 1800 steps (3 minutes) to 1200 steps (2 minutes)
   - This matches the environment's max_episode_time of 120 seconds

2. Updated evaluation loop step limit:
   - Changed from 3000 steps (5 minutes) to 1200 steps (2 minutes)
   - This also matches the environment's max_episode_time of 120 seconds

3. Added debugging information to output:
   - Now shows the environment's internal time tracking alongside step count
   - Displays max_episode_time for reference

4. Fixed comments to reflect the actual 2-minute time limit

These changes ensure that the training and evaluation loops respect the environment's intended 2-minute maximum episode duration.

## How to Test

During training or evaluation, watch the debug output for time-related information:
```
Step 50 | Task: ❌ | Time: 5.1s | Env Time: 5.0s | Max Time: 120s | Reward: -10.5 | Action: Forward
```

The episode should terminate when either:
1. The task is completed successfully
2. The environment time reaches or exceeds 120 seconds
3. The step count reaches 1200 (2 minutes at 10Hz)

## Note
This fix aligns the step limits in the training loops with the environment's max_episode_time, ensuring consistent behavior. Both the environment's internal time tracking and the step limits will now enforce the same 2-minute maximum episode duration.
