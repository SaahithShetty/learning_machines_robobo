#!/usr/bin/env zsh
"""
RL Training Wrapper Script

This script provides a clean interface for training RL agents without exposing 
Docker container internal paths.

Usage:
    ./train_rl.zsh policy_gradient --episodes 300 --learning-rate 0.0005
    ./train_rl.zsh dqn --episodes 500 --batch-size 64
    ./train_rl.zsh qlearning --episodes 1000 --epsilon-decay 0.995
    ./train_rl.zsh actor_critic --episodes 400 --gamma 0.99
"""

set -e

# Check if method is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <method> [training_args...]"
    echo ""
    echo "Available methods:"
    echo "  policy_gradient  - Policy Gradient (REINFORCE)"
    echo "  dqn             - Deep Q-Network"
    echo "  qlearning       - Q-Learning (tabular)"
    echo "  actor_critic    - Actor-Critic (A2C)"
    echo ""
    echo "Examples:"
    echo "  $0 policy_gradient --episodes 300 --learning-rate 0.0005"
    echo "  $0 dqn --episodes 500 --batch-size 64"
    echo "  $0 qlearning --episodes 1000 --epsilon-decay 0.995"
    echo "  $0 actor_critic --episodes 400 --gamma 0.99"
    echo ""
    echo "For more options, run:"
    echo "  $0 <method> --help"
    exit 1
fi

# Extract method and remaining arguments
METHOD=$1
shift
TRAINING_ARGS="$@"

# Validate method
case $METHOD in
    policy_gradient|dqn|qlearning|actor_critic)
        echo "Training $METHOD agent with arguments: $TRAINING_ARGS"
        ;;
    *)
        echo "Error: Unknown method '$METHOD'"
        echo "Available methods: policy_gradient, dqn, qlearning, actor_critic"
        exit 1
        ;;
esac

# Build the Docker image for Apple Silicon
echo "Building Docker image..."
docker build --platform linux/amd64 --tag learning_machines . > /dev/null 2>&1

# Run RL training in Docker container
echo "Starting RL training in Docker container..."
docker run -t --rm --platform linux/amd64 \
    -p 45100:45100 -p 45101:45101 \
    -v "$(pwd)/results:/root/results" \
    -e SCENE_FILE="arena_obstacles.ttt" \
    learning_machines python3 /root/catkin_ws/src/learning_machines/scripts/train_rl.py --method $METHOD $TRAINING_ARGS

echo ""
echo "Training completed! Results saved to ./results/figures/"
