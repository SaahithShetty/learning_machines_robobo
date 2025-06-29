#!/usr/bin/env python3
"""
Training Results Plotter

This script loads saved training metrics from JSON files and generates
comprehensive training plots for Task 3 (Object Pushing to Target).

Usage:
    python plot_training_results.py [json_file_path]
    
If no path is provided, it will look for the most recent training_metrics_*.json file.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path
import glob
from datetime import datetime

def load_training_data(json_path=None):
    """Load training data from JSON file"""
    if json_path is None:
        # Find the most recent training metrics file
        results_dir = Path("results/figures")
        if not results_dir.exists():
            results_dir = Path("examples/full_project_setup/results/figures")
        
        if results_dir.exists():
            json_files = list(results_dir.glob("training_metrics_*.json"))
            if json_files:
                # Get the most recent file
                json_path = max(json_files, key=lambda f: f.stat().st_mtime)
                print(f"Using most recent training file: {json_path}")
            else:
                print("No training_metrics_*.json files found!")
                return None
        else:
            print("Results directory not found!")
            return None
    
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        print(f"Loaded training data: {data['episodes']} episodes, agent: {data['agent_type'].upper()}")
        return data
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return None

def create_training_plots(results, output_path=None):
    """Create comprehensive training plots from results data"""
    if not results or 'episode_rewards' not in results:
        print("Invalid results data - missing episode_rewards")
        return None
    
    agent_type = results.get('agent_type', 'unknown')
    num_episodes = results.get('episodes', len(results['episode_rewards']))
    
    # Set up the figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Task 3 Training Results - {agent_type.upper()} ({num_episodes} episodes)', fontsize=16)
    
    # 1. Episode Rewards
    rewards = np.array(results['episode_rewards'])/4
    episodes = np.arange(1, len(rewards) + 1)
    
    axes[0, 0].plot(episodes, rewards, 'b-', alpha=0.7, linewidth=1)
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Add statistics
    avg_reward = np.mean(rewards)
    max_reward = np.max(rewards)
    min_reward = np.min(rewards)
    axes[0, 0].axhline(y=avg_reward, color='r', linestyle='--', alpha=0.8, label=f'Avg: {avg_reward:.1f}')
    axes[0, 0].legend()
    
    # 2. Moving Average Rewards
    window = min(50, len(rewards) // 10)  # Use appropriate window size
    if window > 1:
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        ma_episodes = np.arange(window, len(rewards) + 1)
        
        axes[0, 1].plot(ma_episodes, moving_avg, 'r-', linewidth=2, label=f'MA({window})')
        axes[0, 1].plot(episodes, rewards, 'b-', alpha=0.3, linewidth=0.5, label='Raw rewards')
        axes[0, 1].set_title(f'Moving Average Rewards (window={window})')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Average Reward')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()
    
    # 3. Task Completion Statistics
    if 'episode_task_completions' in results:
        completions = results['episode_task_completions']
        axes[1, 0].plot(episodes[:len(completions)], completions, 'g-', linewidth=2)
        axes[1, 0].axhline(y=1, color='r', linestyle='--', alpha=0.8, label='Target (Task Complete)')
        axes[1, 0].set_title('Task Completion per Episode')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Task Completed (1=Yes, 0=No)')
        axes[1, 0].set_ylim(-0.1, 1.3)  # Increased upper limit to give more space
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()
        
        # Add success rate text
        success_episodes = sum(completions)
        success_rate = (success_episodes / len(completions)) * 100
        axes[1, 0].text(0.02, 0.98, f'Success Rate: {success_rate:.1f}%', 
                        transform=axes[1, 0].transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    else:
        axes[1, 0].text(0.5, 0.5, 'Task completion data\nnot available', 
                        transform=axes[1, 0].transAxes, ha='center', va='center')
        axes[1, 0].set_title('Task Completion (No Data)')
    
    # 4. Episode Length / Success Rate
    if 'episode_success_rates' in results:
        success_rates = results['episode_success_rates']
        # Calculate running success rate (last N episodes)
        window_size = min(20, len(success_rates) // 5)
        if window_size > 1:
            running_success = []
            for i in range(len(success_rates)):
                start_idx = max(0, i - window_size + 1)
                window_data = success_rates[start_idx:i+1]
                running_success.append(np.mean(window_data) * 100)
            
            axes[1, 1].plot(episodes[:len(running_success)], running_success, 'purple', linewidth=2)
            axes[1, 1].set_title(f'Running Success Rate (window={window_size})')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Success Rate (%)')
            axes[1, 1].set_ylim(0, 100)
            axes[1, 1].grid(True, alpha=0.3)
    elif 'episode_lengths' in results:
        lengths = results['episode_lengths']
        axes[1, 1].plot(episodes[:len(lengths)], lengths, 'm-', linewidth=1)
        axes[1, 1].set_title('Episode Lengths (Steps)')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Steps per Episode')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add average line
        avg_length = np.mean(lengths)
        axes[1, 1].axhline(y=avg_length, color='r', linestyle='--', alpha=0.8, label=f'Avg: {avg_length:.1f}')
        axes[1, 1].legend()
    else:
        axes[1, 1].text(0.5, 0.5, 'Episode length/success\ndata not available', 
                        transform=axes[1, 1].transAxes, ha='center', va='center')
        axes[1, 1].set_title('Episode Statistics (No Data)')
    
    # Add summary statistics
    summary_text = f"""Training Summary:
    Episodes: {num_episodes}
    Avg Reward: {avg_reward:.1f}
    Best Reward: {max_reward:.1f}
    Worst Reward: {min_reward:.1f}"""
    
    if 'episode_task_completions' in results:
        completions = results['episode_task_completions']
        success_episodes = sum(completions)
        success_rate = (success_episodes / len(completions)) * 100
        summary_text += f"\nSuccess Rate: {success_rate:.1f}%\nCompleted: {success_episodes}/{len(completions)}"
    
    if 'episode_lengths' in results:
        lengths = results['episode_lengths']
        avg_length = np.mean(lengths)
        summary_text += f"\nAvg Length: {avg_length:.1f} steps"
    
    fig.text(0.02, 0.02, summary_text, fontsize=10, 
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    
    # Save the plot
    if output_path is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = f"training_plot_{agent_type}_{timestamp}.png"
    
    # Save the combined plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Combined training plot saved to: {output_path}")
    
    # Save individual plots
    base_name = Path(output_path).stem
    output_dir = Path(output_path).parent
    
    # Save Episode Rewards plot
    plt.figure(figsize=(8, 6))
    plt.plot(episodes, rewards, 'b-', alpha=0.7, linewidth=1)
    plt.title(f'Episode Rewards - {agent_type.upper()}')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=avg_reward, color='r', linestyle='--', alpha=0.8, label=f'Avg: {avg_reward:.1f}')
    plt.legend()
    rewards_path = output_dir / f"{base_name}_rewards.png"
    plt.savefig(rewards_path, dpi=300, bbox_inches='tight')
    print(f"Episode rewards plot saved to: {rewards_path}")
    plt.close()
    
    # Save Moving Average plot if available
    if window > 1 and 'moving_avg' in locals() and 'ma_episodes' in locals():
        plt.figure(figsize=(8, 6))
        plt.plot(ma_episodes, moving_avg, 'r-', linewidth=2, label=f'MA({window})')
        plt.plot(episodes, rewards, 'b-', alpha=0.3, linewidth=0.5, label='Raw rewards')
        plt.title(f'Moving Average Rewards - {agent_type.upper()}')
        plt.xlabel('Episode')
        plt.ylabel('Average Reward')
        plt.grid(True, alpha=0.3)
        plt.legend()
        ma_path = output_dir / f"{base_name}_moving_avg.png"
        plt.savefig(ma_path, dpi=300, bbox_inches='tight')
        print(f"Moving average plot saved to: {ma_path}")
        plt.close()
    
    # Save Task Completion plot
    if 'episode_task_completions' in results:
        completions = results['episode_task_completions']
        success_episodes = sum(completions)
        success_rate = (success_episodes / len(completions)) * 100
        
        plt.figure(figsize=(8, 6))
        plt.plot(episodes[:len(completions)], completions, 'g-', linewidth=2)
        plt.axhline(y=1, color='r', linestyle='--', alpha=0.8, label='Target (Task Complete)')
        plt.title(f'Task Completion - {agent_type.upper()}')
        plt.xlabel('Episode')
        plt.ylabel('Task Completed (1=Yes, 0=No)')
        plt.ylim(-0.1, 1.3)  # Increased upper limit to give more space
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.text(0.02, 0.98, f'Success Rate: {success_rate:.1f}%', 
                 transform=plt.gca().transAxes, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        completion_path = output_dir / f"{base_name}_completions.png"
        plt.savefig(completion_path, dpi=300, bbox_inches='tight')
        print(f"Task completion plot saved to: {completion_path}")
        plt.close()
    
    # Save Success Rate or Episode Length plot
    if 'episode_success_rates' in results:
        success_rates = results['episode_success_rates']
        window_size = min(20, len(success_rates) // 5)
        
        if window_size > 1:
            running_success = []
            for i in range(len(success_rates)):
                start_idx = max(0, i - window_size + 1)
                window_data = success_rates[start_idx:i+1]
                running_success.append(np.mean(window_data) * 100)
            
            plt.figure(figsize=(8, 6))
            plt.plot(episodes[:len(running_success)], running_success, 'purple', linewidth=2)
            plt.title(f'Running Success Rate - {agent_type.upper()}')
            plt.xlabel('Episode')
            plt.ylabel('Success Rate (%)')
            plt.ylim(0, 100)
            plt.grid(True, alpha=0.3)
            success_path = output_dir / f"{base_name}_success_rate.png"
            plt.savefig(success_path, dpi=300, bbox_inches='tight')
            print(f"Success rate plot saved to: {success_path}")
            plt.close()
    
    elif 'episode_lengths' in results:
        lengths = results['episode_lengths']
        avg_length = np.mean(lengths)
        
        plt.figure(figsize=(8, 6))
        plt.plot(episodes[:len(lengths)], lengths, 'm-', linewidth=1)
        plt.title(f'Episode Lengths - {agent_type.upper()}')
        plt.xlabel('Episode')
        plt.ylabel('Steps per Episode')
        plt.grid(True, alpha=0.3)
        plt.axhline(y=avg_length, color='r', linestyle='--', alpha=0.8, label=f'Avg: {avg_length:.1f}')
        plt.legend()
        length_path = output_dir / f"{base_name}_lengths.png"
        plt.savefig(length_path, dpi=300, bbox_inches='tight')
        print(f"Episode length plot saved to: {length_path}")
        plt.close()
    
    # Show the combined plot
    plt.figure(fig.number)
    plt.show()
    
    return output_path

def main():
    """Main function"""
    # Check command line arguments
    json_path = None
    if len(sys.argv) > 1:
        json_path = sys.argv[1]
        if not Path(json_path).exists():
            print(f"File not found: {json_path}")
            return
    
    # Load training data
    results = load_training_data(json_path)
    if results is None:
        return
    
    # Create and save plots
    output_path = create_training_plots(results)
    if output_path:
        print(f"✅ Plot generation completed!")
        print(f"📊 View your training results at: {output_path}")

if __name__ == "__main__":
    main()
