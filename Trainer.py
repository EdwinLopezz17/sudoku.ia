from SudokuEnv import SudokuEnv
from DQNAgent import DQNAgent

import numpy as np
import random
import time
import os
import matplotlib.pyplot as plt

# Training parameters
START_FROM = 1801
EPISODES = 3000
MAX_STEPS = 81
BATCH_SIZE = 64
UPDATE_TARGET_EVERY = 10  # Update target network every N episodes
SAVE_MODEL_EVERY = 100    # Save model every N episodes
LOG_EVERY = 5            # Log detailed metrics every N episodes

# Create directories if they don't exist
os.makedirs('models', exist_ok=True)
os.makedirs('logs', exist_ok=True)

# Initialize puzzles with varying difficulties
puzzles = {
    'easy': [
        np.array([
            [0, 4, 2, 0, 0, 5, 0, 0, 6],
            [1, 9, 7, 0, 0, 0, 0, 4, 0],
            [5, 6, 0, 4, 0, 0, 1, 0, 9],
            [8, 0, 1, 3, 0, 0, 2, 6, 0],
            [9, 0, 0, 7, 1, 0, 4, 5, 0],
            [0, 3, 2, 5, 6, 0, 0, 0, 0],
            [0, 5, 3, 0, 2, 7, 0, 0, 0],
            [4, 5, 9, 0, 0, 6, 0, 0, 0],
            [0, 7, 6, 0, 0, 0, 0, 0, 8]
        ])
    ]
}


# Initialize environment and agent
env = SudokuEnv()
agent = DQNAgent()

#cargar modelo aterior
agent.load("models/model_sudoku_ia_ep1800.keras") #modelo anterior
agent.epsilon = 0.1
print("✅ Modelo cargado, continuando entrenamiento...")


#metrics tracking
all_rewards = []
valid_moves_per_episode = []
completed_rows = []
completed_cols = []
completed_boxes = []
solved_puzzles = []

def plot_metrics(episode):
    plt.figure(figsize=(15, 10))
    
    # Plot rewards
    plt.subplot(2, 2, 1)
    plt.plot(all_rewards)
    plt.title('Total Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    
    # Plot valid moves ratio
    plt.subplot(2, 2, 2)
    plt.plot(valid_moves_per_episode)
    plt.title('Valid Moves Ratio per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Valid Moves / Total Moves')
    plt.ylim(0, 1.0)
    
    # Plot completed constraints
    plt.subplot(2, 2, 3)
    plt.plot(completed_rows, label='Rows')
    plt.plot(completed_cols, label='Columns')
    plt.plot(completed_boxes, label='Boxes')
    plt.title('Completed Constraints per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Count')
    plt.legend()
    
    # Plot solved puzzles (cumulative)
    plt.subplot(2, 2, 4)
    plt.plot(np.cumsum(solved_puzzles))
    plt.title('Cumulative Solved Puzzles')
    plt.xlabel('Episode')
    plt.ylabel('Count')
    
    plt.tight_layout()
    plt.savefig(f'logs/metrics_ep{episode}.png')
    plt.close()

for e in range(START_FROM, EPISODES + 1):
    # Select puzzle based on agent's performance
    # Initially start with easy puzzles, then gradually increase difficulty
    if e < 500:
        puzzle_key = 'easy'
    else:
        puzzle_key = 'easy'  # Change to more difficult puzzles later

    puzzle = np.array(random.choice(puzzles[puzzle_key])).reshape((9, 9))

    print(f"\n🔁 === Starting Episode {e}/{EPISODES} ({puzzle_key} puzzle) ===")
    state = env.reset(puzzle)
    done = False
    total_reward = 0
    steps = 0
    valid_moves = 0
    invalid_moves = 0
    
    # Track completed constraints
    episode_rows_completed = 0
    episode_cols_completed = 0
    episode_boxes_completed = 0
    puzzle_solved = 0
    
    while not done and steps < MAX_STEPS:
        # Get the action mask and constraint info
        action_mask = env.get_action_mask()
        constraints = env.get_number_constraints()
        
        # Choose an action
        action_index = agent.act(state, action_mask, constraints)
        row, col, num = agent.decode_action(action_index)
        
        # Take the action
        next_state, reward, done, info = env.step((row, col, num))
        
        # Get the next action mask and constraint info
        next_action_mask = env.get_action_mask()
        next_constraints = env.get_number_constraints()
        
        # Track metrics
        if info.get('valid_move', False):
            valid_moves += 1
        else:
            invalid_moves += 1
            
        if info.get('row_complete', False):
            episode_rows_completed += 1
        if info.get('col_complete', False):
            episode_cols_completed += 1
        if info.get('box_complete', False):
            episode_boxes_completed += 1
        if info.get('solved', False):
            puzzle_solved = 1
        
        # Remember experience
        agent.remember(state, action_index, reward, next_state, done, 
                       action_mask, next_action_mask, constraints, next_constraints)
        
        # Update state and metrics
        state = next_state
        total_reward += reward
        steps += 1
        
        # Learn from experience
        agent.replay(batch_size=BATCH_SIZE)
    
    # Update target network periodically
    if e % UPDATE_TARGET_EVERY == 0:
        agent.update_target_model()
        print(f"🎯 Target network updated at episode {e}")
    
    # Log metrics
    all_rewards.append(total_reward)
    valid_moves_per_episode.append(valid_moves / max(1, steps))
    completed_rows.append(episode_rows_completed)
    completed_cols.append(episode_cols_completed)
    completed_boxes.append(episode_boxes_completed)
    solved_puzzles.append(puzzle_solved)
    
    # Print episode summary
    print(f"🧠 Episode {e}/{EPISODES}")
    print(f"   ➕ Total Reward     : {total_reward}")
    print(f"   🔄 Steps Executed   : {steps}/{MAX_STEPS}")
    print(f"   ✅ Valid Moves      : {valid_moves}/{steps} ({valid_moves/max(1, steps):.2%})")
    print(f"   ❌ Invalid Moves    : {invalid_moves}")
    print(f"   🏆 Completed Rows   : {episode_rows_completed}")
    print(f"   🏆 Completed Columns: {episode_cols_completed}")
    print(f"   🏆 Completed Boxes  : {episode_boxes_completed}")
    print(f"   🎯 Epsilon          : {agent.epsilon:.3f}")
    
    if puzzle_solved:
        print(f"   🌟 PUZZLE SOLVED! 🌟")
        
    # Print the board state every LOG_EVERY episodes
    if e % LOG_EVERY == 0:
        print("\nCurrent board state:")
        env.render()
    
    # Save model periodically
    if e % SAVE_MODEL_EVERY == 0:
        model_path = f"models/model_sudoku_ia_ep{e}.keras"

        agent.save(model_path)
        
        print(f"💾 Model saved: {model_path}")
        
        # Plot and save metrics
        plot_metrics(e)

# Final metrics plot
plot_metrics(EPISODES)
print("\n✅ Training completed!")

# Test the final model on a new puzzle
print("\n🧪 Testing final model on a new puzzle...")
test_puzzle = puzzles['easy'][0]  # Use the first easy puzzle for testing
state = env.reset(test_puzzle)
print("Initial state:")
env.render()

done = False
steps = 0
while not done and steps < MAX_STEPS:
    # Get action mask and constraints
    action_mask = env.get_action_mask()
    constraints = env.get_number_constraints()
    
    # Choose action (no exploration in testing)
    agent.epsilon = 0
    action_index = agent.act(state, action_mask, constraints)
    row, col, num = agent.decode_action(action_index)
    
    # Take action
    state, reward, done, info = env.step((row, col, num))
    steps += 1
    
    if steps % 10 == 0 or done:
        print(f"After {steps} steps:")
        env.render()

print("Final state after testing:")
env.render()

if env.check_done():
    print("🌟 Test puzzle solved successfully! 🌟")
else:
    print("❌ Test puzzle was not solved completely.")


