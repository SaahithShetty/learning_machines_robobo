#!/usr/bin/env zsh

set -e

# Function to show usage
show_usage() {
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  train <method> [args...]     - Train RL agent (simulation)"
    echo "  train-hw <method> [args...]  - Train RL agent (hardware)"
    echo "  baseline <method> [args...]  - Run baseline algorithm (simulation)"
    echo "  baseline-hw <method> [args...] - Run baseline algorithm (hardware)"
    echo "  run [args...]                - Run custom command"
    echo "  rebuild                      - Force rebuild Docker image"
    echo "  skip-build <command>         - Skip build and run command"
    echo ""
    echo "RL Training Methods:"
    echo "  policy_gradient          - Policy Gradient (REINFORCE)"
    echo "  dqn                      - Deep Q-Network"
    echo "  qlearning                - Q-Learning (tabular)"
    echo "  actor_critic             - Actor-Critic (A2C)"
    echo ""
    echo "Baseline Methods:"
    echo "  obstacle_avoidance       - Advanced rule-based obstacle avoidance"
    echo "  wall_following           - Wall-following algorithm"
    echo ""
    echo "Examples:"
    echo "  $0 train policy_gradient --episodes 300 --learning-rate 0.0005"
    echo "  $0 train-hw dqn --episodes 500 --batch-size 64"
    echo "  $0 baseline obstacle_avoidance --duration 120 --max-distance 10"
    echo "  $0 baseline-hw wall_following --duration 60 --wall-distance 0.3"
    echo "  $0 train qlearning --episodes 1000 --epsilon-decay 0.995"
    echo "  $0 train actor_critic --episodes 400 --gamma 0.99"
    echo ""
    echo "  $0 run python3 your_script.py"
    echo "  $0 run bash"
    echo "  $0 rebuild  # Force rebuild Docker image"
    echo "  $0 skip-build train policy_gradient --episodes 50  # Skip build for faster testing"
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
            policy_gradient|dqn|qlearning|actor_critic)
                echo "Training $METHOD agent (simulation) with arguments: $TRAINING_ARGS"
                DOCKER_CMD="python3 /root/catkin_ws/src/learning_machines/scripts/train_rl.py --method $METHOD --simulation $TRAINING_ARGS"
                ;;
            *)
                echo "Error: Unknown RL method '$METHOD'"
                echo "Available methods: policy_gradient, dqn, qlearning, actor_critic"
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
            policy_gradient|dqn|qlearning|actor_critic)
                echo "Training $METHOD agent (hardware) with arguments: $TRAINING_ARGS"
                DOCKER_CMD="python3 /root/catkin_ws/src/learning_machines/scripts/train_rl.py --method $METHOD --hardware $TRAINING_ARGS"
                ;;
            *)
                echo "Error: Unknown RL method '$METHOD'"
                echo "Available methods: policy_gradient, dqn, qlearning, actor_critic"
                exit 1
                ;;
        esac
        ;;
    baseline)
        if [ $# -eq 0 ]; then
            echo "Error: No baseline method specified"
            echo ""
            show_usage
            exit 1
        fi
        
        METHOD=$1
        shift
        BASELINE_ARGS="$@"
        
        # Validate baseline method
        case $METHOD in
            obstacle_avoidance|wall_following)
                echo "Running baseline $METHOD algorithm (simulation) with arguments: $BASELINE_ARGS"
                if [ "$METHOD" = "obstacle_avoidance" ]; then
                    DOCKER_CMD="python3 /root/catkin_ws/src/learning_machines/scripts/task0_controller.py --simulation --method obstacle_avoidance_task1 $BASELINE_ARGS"
                else
                    DOCKER_CMD="python3 /root/catkin_ws/src/learning_machines/scripts/task0_controller.py --simulation --method wall_following_algorithm $BASELINE_ARGS"
                fi
                ;;
            *)
                echo "Error: Unknown baseline method '$METHOD'"
                echo "Available methods: obstacle_avoidance, wall_following"
                exit 1
                ;;
        esac
        ;;
    baseline-hw)
        if [ $# -eq 0 ]; then
            echo "Error: No baseline method specified"
            echo ""
            show_usage
            exit 1
        fi
        
        METHOD=$1
        shift
        BASELINE_ARGS="$@"
        
        # Validate baseline method
        case $METHOD in
            obstacle_avoidance|wall_following)
                echo "Running baseline $METHOD algorithm (hardware) with arguments: $BASELINE_ARGS"
                if [ "$METHOD" = "obstacle_avoidance" ]; then
                    DOCKER_CMD="python3 /root/catkin_ws/src/learning_machines/scripts/task0_controller.py --hardware --method obstacle_avoidance_task1 $BASELINE_ARGS"
                else
                    DOCKER_CMD="python3 /root/catkin_ws/src/learning_machines/scripts/task0_controller.py --hardware --method wall_following_algorithm $BASELINE_ARGS"
                fi
                ;;
            *)
                echo "Error: Unknown baseline method '$METHOD'"
                echo "Available methods: obstacle_avoidance, wall_following"
                exit 1
                ;;
        esac
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

# Run the container with obstacle environment
echo "Starting Docker container..."
docker run -t --rm --platform linux/amd64 \
    -p 45100:45100 -p 45101:45101 \
    -v "$(pwd)/results:/root/results" \
    -e SCENE_FILE="arena_obstacles.ttt" \
    learning_machines $DOCKER_CMD

echo ""
echo "Execution completed! Results saved to ./results/"
