#!/usr/bin/env zsh

set -e

# Function to show usage
show_usage() {
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  train <method> [args...]     - Train RL agent for Task 3 (simulation)"
    echo "  train-hw <method> [args...]  - Train RL agent for Task 3 (hardware)"
    echo "  test <method> [args...]      - Test/evaluate RL agent (simulation)"
    echo "  test-hw <method> [args...]   - Test/evaluate RL agent (hardware)"
    echo "  task3 [args...]              - Run Task 3 controller directly"
    echo "  monitor [args...]            - Monitor IR sensors for debugging"
    echo "  run [args...]                - Run custom command"
    echo "  rebuild                      - Force rebuild Docker image"
    echo "  skip-build <command>         - Skip build and run command"
    echo ""
    echo "RL Methods (Task 3: Object Pushing):"
    echo "  dqn                      - Deep Q-Network (recommended)"
    echo "  qlearning                - Q-Learning (tabular)"
    echo "  policy_gradient          - Policy Gradient (REINFORCE)"
    echo "  actor_critic             - Actor-Critic (A2C)"
    echo ""
    echo "Examples (Task 3: Object Pushing):"
    echo "  $0 train dqn --episodes 100"
    echo "  $0 train-hw dqn --episodes 50 --mode train_and_evaluate"
    echo "  $0 test dqn --episodes 10 --load-model /root/results/rl_model_dqn_*.pth"
    echo "  $0 train qlearning --episodes 200"
    echo "  $0 train policy_gradient --episodes 150"
    echo "  $0 train actor_critic --episodes 100"
    echo ""
    echo "Advanced options:"
    echo "  $0 train dqn --episodes 50 --collision-threshold 0.95  # Custom threshold"
    echo "  $0 train dqn --episodes 50 --use-thresholds  # Legacy threshold mode (NOT recommended)"
    echo ""
    echo "Direct Task 3 access:"
    echo "  $0 task3 --simulation --method dqn --episodes 50"
    echo "  $0 monitor --simulation  # Monitor IR sensors"
    echo ""
    echo "Custom commands:"
    echo "  $0 run python3 /root/catkin_ws/src/learning_machines/scripts/learning_robobo_controller.py --simulation --method dqn"
    echo "  $0 run bash"
    echo "  $0 rebuild  # Force rebuild Docker image"
    echo "  $0 skip-build train dqn --episodes 50  # Skip build for faster testing"
}

# Check if any arguments provided
if [ $# -eq 0 ]; then
    show_usage
    exit 1
fi

# Parse command
COMMAND=$1
shift

case $COMMAND in
    train)
        if [ $# -eq 0 ]; then
            echo "Error: No RL method specified"
            echo ""
            show_usage
            exit 1
        fi
        
        METHOD=$1
        shift
        TRAINING_ARGS="$@"
        
        # Validate RL method
        case $METHOD in
            dqn|qlearning|policy_gradient|actor_critic)
                echo "Training $METHOD agent for Task 3 (simulation) with arguments: $TRAINING_ARGS"
                DOCKER_CMD="python3 /root/catkin_ws/src/learning_machines/scripts/train_rl.py --simulation --method $METHOD $TRAINING_ARGS"
                ;;
            *)
                echo "Error: Unknown RL method '$METHOD'"
                echo "Available methods: dqn, qlearning, policy_gradient, actor_critic"
                exit 1
                ;;
        esac
        ;;
    train-hw)
        if [ $# -eq 0 ]; then
            echo "Error: No RL method specified"
            echo ""
            show_usage
            exit 1
        fi
        
        METHOD=$1
        shift
        TRAINING_ARGS="$@"
        
        # Validate RL method
        case $METHOD in
            dqn|qlearning|policy_gradient|actor_critic)
                echo "Training $METHOD agent for Task 3 (hardware) with arguments: $TRAINING_ARGS"
                DOCKER_CMD="python3 /root/catkin_ws/src/learning_machines/scripts/train_rl.py --hardware --method $METHOD $TRAINING_ARGS"
                ;;
            *)
                echo "Error: Unknown RL method '$METHOD'"
                echo "Available methods: dqn, qlearning, policy_gradient, actor_critic"
                exit 1
                ;;
        esac
        ;;
    test)
        if [ $# -eq 0 ]; then
            echo "Error: No RL method specified"
            echo ""
            show_usage
            exit 1
        fi
        
        METHOD=$1
        shift
        TEST_ARGS="$@"
        
        # Validate RL method
        case $METHOD in
            dqn|qlearning|policy_gradient|actor_critic)
                echo "Testing $METHOD agent for Task 3 (simulation) with arguments: $TEST_ARGS"
                DOCKER_CMD="python3 /root/catkin_ws/src/learning_machines/scripts/train_rl.py --simulation --method $METHOD --mode evaluate $TEST_ARGS"
                ;;
            *)
                echo "Error: Unknown RL method '$METHOD'"
                echo "Available methods: dqn, qlearning, policy_gradient, actor_critic"
                exit 1
                ;;
        esac
        ;;
    test-hw)
        if [ $# -eq 0 ]; then
            echo "Error: No RL method specified"
            echo ""
            show_usage
            exit 1
        fi
        
        METHOD=$1
        shift
        TEST_ARGS="$@"
        
        # Validate RL method
        case $METHOD in
            dqn|qlearning|policy_gradient|actor_critic)
                echo "Testing $METHOD agent for Task 3 (hardware) with arguments: $TEST_ARGS"
                DOCKER_CMD="python3 /root/catkin_ws/src/learning_machines/scripts/train_rl.py --hardware --method $METHOD --mode evaluate $TEST_ARGS"
                ;;
            *)
                echo "Error: Unknown RL method '$METHOD'"
                echo "Available methods: dqn, qlearning, policy_gradient, actor_critic"
                exit 1
                ;;
        esac
        ;;
    task3)
        # Direct access to Task 3 controller
        TASK3_ARGS="$@"
        echo "Running Task 3 controller with arguments: $TASK3_ARGS"
        DOCKER_CMD="python3 /root/catkin_ws/src/learning_machines/scripts/task3_controller.py $TASK3_ARGS"
        ;;
    monitor)
        # Monitor IR sensors for debugging
        MONITOR_ARGS="$@"
        echo "Running IR sensor monitor with arguments: $MONITOR_ARGS"
        DOCKER_CMD="python3 /root/catkin_ws/src/learning_machines/scripts/monitor_ir_sensors.py $MONITOR_ARGS"
        ;;
    rebuild)
        echo "Force rebuilding Docker image..."
        docker build --platform linux/amd64 --tag learning_machines .
        echo "Docker image rebuilt successfully!"
        exit 0
        ;;
    skip-build)
        echo "Skipping Docker build (using existing image)..."
        SKIP_BUILD=true
        COMMAND=$1
        shift
        ;;
    run)
        DOCKER_CMD="$@"
        echo "Running custom command: $DOCKER_CMD"
        ;;
    --help|-h|help)
        show_usage
        exit 0
        ;;
    *)
        # Backward compatibility - treat first argument as direct command
        DOCKER_CMD="$COMMAND $@"
        echo "Running command: $DOCKER_CMD"
        ;;
esac

# Build the Docker image for Apple Silicon
if [ "$SKIP_BUILD" != "true" ]; then
    echo "Building Docker image..."
    docker build --platform linux/amd64 --tag learning_machines .
else
    echo "Skipping build (using existing image)..."
fi

# Run the container with Task 3 environment (object pushing)
echo "Starting Docker container for Task 3: Object Pushing..."
docker run -it --rm --platform linux/amd64 \
    -p 45100:45100 -p 45101:45101 \
    -v "$(pwd)/results:/root/results" \
    -e SCENE_FILE="arena_approach.ttt" \
    learning_machines $DOCKER_CMD

echo ""
echo "Execution completed! Results saved to ./results/"
