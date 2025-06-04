import gym
import numpy as np
import time
import matplotlib.pyplot as plt
from collections import deque

env = gym.make("CartPole-v1", render_mode="human")
obs, info = env.reset()

learning_rate = 0.05
discount_factor = 0.99
exploration_rate = 1.0
exploration_decay = 0.998
min_exploration_rate = 0.01
num_episodes = 1000

num_bins = 20
observation_space_high = env.observation_space.high
observation_space_low = env.observation_space.low

observation_space_high[1] = 4
observation_space_high[3] = 4
observation_space_low[1] = -4
observation_space_low[3] = -4

position_bins = 16
velocity_bins = 20
angle_bins = 24
ang_velocity_bins = 20

def get_discrete_state(state):
    position_adj = (state[0] - observation_space_low[0]) * position_bins / (observation_space_high[0] - observation_space_low[0])
    velocity_adj = (state[1] - observation_space_low[1]) * velocity_bins / (observation_space_high[1] - observation_space_low[1])
    angle_adj = (state[2] - observation_space_low[2]) * angle_bins / (observation_space_high[2] - observation_space_low[2])
    ang_vel_adj = (state[3] - observation_space_low[3]) * ang_velocity_bins / (observation_space_high[3] - observation_space_low[3])
    
    position_idx = min(position_bins - 1, max(0, int(position_adj)))
    velocity_idx = min(velocity_bins - 1, max(0, int(velocity_adj)))
    angle_idx = min(angle_bins - 1, max(0, int(angle_adj)))
    ang_vel_idx = min(ang_velocity_bins - 1, max(0, int(ang_vel_adj)))
    
    return (position_idx, velocity_idx, angle_idx, ang_vel_idx)

q_table_file = 'trained_q_table.npy'
try:
    print(f"Attempting to load existing Q-table from {q_table_file}...")
    q_table = np.load(q_table_file)
    print(f"Loaded existing Q-table with shape {q_table.shape}")
    expected_shape = (position_bins, velocity_bins, angle_bins, ang_velocity_bins, env.action_space.n)
    if q_table.shape != expected_shape:
        print(f"Warning: Loaded Q-table has shape {q_table.shape}, expected {expected_shape}")
        print("Creating a new Q-table instead.")
        q_table = np.zeros(expected_shape)
except FileNotFoundError:
    print("No existing Q-table found. Creating a new one.")
    q_table = np.zeros([position_bins, velocity_bins, angle_bins, ang_velocity_bins, env.action_space.n])

total_rewards = []
avg_rewards = deque(maxlen=100)
best_reward = 0
best_episode = 0

for episode in range(num_episodes):
    obs, info = env.reset()
    discrete_state = get_discrete_state(obs)
    done = False
    episode_reward = 0
    step_count = 0
    
    while not done:
        if np.random.random() < exploration_rate:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[discrete_state])
        
        new_obs, reward, terminated, truncated, info = env.step(action)
        discrete_new_state = get_discrete_state(new_obs)
        done = terminated or truncated
        
        position = new_obs[0]
        angle = new_obs[2]
        
        if done and step_count < 100:
            reward = -1
        
        episode_reward += reward
        step_count += 1
        
        print(f"\rEpisode: {episode}/{num_episodes} | Steps: {step_count} | Reward: {episode_reward} | " 
              f"Exploration: {exploration_rate:.2f} | Position: {new_obs[0]:.2f} | Angle: {new_obs[2]:.2f}", end="")
        
        if not done:
            max_future_q = np.max(q_table[discrete_new_state])
            current_q = q_table[discrete_state + (action,)]
            new_q = current_q + learning_rate * (reward + discount_factor * max_future_q - current_q)
            q_table[discrete_state + (action,)] = new_q
        else:
            q_table[discrete_state + (action,)] = 0
        
        discrete_state = discrete_new_state
        obs = new_obs
        
        time.sleep(0.01)
    
    total_rewards.append(episode_reward)
    avg_rewards.append(episode_reward)
    
    if episode_reward > best_reward:
        best_reward = episode_reward
        best_episode = episode
    
    print(f" | Avg100: {np.mean(avg_rewards):.2f} | Best: {best_reward} (ep {best_episode})")
    
    exploration_rate = max(min_exploration_rate, exploration_rate * exploration_decay)

np.save('trained_q_table.npy', q_table)
print("Q-table saved to 'trained_q_table.npy'")
history_file = 'training_history.npz'
try:
    history = np.load(history_file, allow_pickle=True)
    prev_rewards = history['rewards'].tolist()
    prev_episodes = history['episodes']
    
    all_rewards = prev_rewards + total_rewards
    all_episodes = prev_episodes + len(total_rewards)
    
    print(f"Updated training history: {prev_episodes} previous + {len(total_rewards)} new episodes = {all_episodes} total")
except FileNotFoundError:
    all_rewards = total_rewards
    all_episodes = len(total_rewards)
    print(f"Created new training history with {all_episodes} episodes")

np.savez(history_file, rewards=all_rewards, episodes=all_episodes)
print(f"Training history saved to {history_file}")

env.close()
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
plt.plot(total_rewards, alpha=0.6, color='steelblue', label='Episode reward')
running_avg = np.convolve(total_rewards, np.ones(100)/100, mode='valid')
plt.plot(running_avg, linewidth=2, color='orangered', label='100-episode average')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Training Progress')
plt.legend()
plt.grid(True)
exploration_rates = [1.0 * (exploration_decay ** i) for i in range(num_episodes)]
exploration_rates = [max(min_exploration_rate, rate) for rate in exploration_rates]
plt.subplot(2, 2, 2)
plt.plot(exploration_rates, color='green')
plt.xlabel('Episode')
plt.ylabel('Exploration Rate')
plt.title('Exploration Rate Decay')
plt.grid(True)
plt.subplot(2, 2, 3)
plt.hist(total_rewards, bins=20, color='purple', alpha=0.7)
plt.axvline(np.mean(total_rewards), color='red', linestyle='dashed', linewidth=2, 
           label=f'Mean: {np.mean(total_rewards):.1f}')
plt.axvline(best_reward, color='green', linestyle='dashed', linewidth=2,
           label=f'Best: {best_reward}')
plt.xlabel('Reward')
plt.ylabel('Frequency')
plt.title('Reward Distribution')
plt.legend()
plt.grid(True)
segment_size = 100
if len(total_rewards) >= segment_size:
    segments = len(total_rewards) // segment_size
    segment_means = [np.mean(total_rewards[i*segment_size:(i+1)*segment_size]) 
                    for i in range(segments)]
    plt.subplot(2, 2, 4)
    plt.bar(range(segments), segment_means, color='teal', alpha=0.7)
    plt.xlabel(f'Training Segment (each {segment_size} episodes)')
    plt.ylabel('Average Reward')
    plt.title('Learning Progress by Segment')
    plt.grid(True, axis='y')

plt.tight_layout()
plt.savefig('enhanced_training_analysis.png', dpi=300)
plt.show()
print("\n" + "="*50)
print("TRAINING SUMMARY")
print("="*50)
print(f"Total episodes: {num_episodes}")
print(f"Best episode: {best_episode} with reward {best_reward}")
print(f"Final exploration rate: {exploration_rate:.4f}")
print(f"Average reward (last 100 episodes): {np.mean(list(avg_rewards)):.2f}")
print(f"Average reward (all episodes): {np.mean(total_rewards):.2f}")
if len(total_rewards) >= 100:
    print(f"Training progression: {np.mean(total_rewards[-100:]) - np.mean(total_rewards[:100]):.2f}")
print("="*50)

print(f"Training visualization saved as 'enhanced_training_analysis.png'")

def test_trained_model(model_path='trained_q_table.npy', episodes=10):
    print("\n" + "="*50)
    print("TESTING TRAINED MODEL")
    print("="*50)
    
    q_table = np.load(model_path)
    
    test_env = gym.make("CartPole-v1", render_mode="human")
    
    for ep in range(episodes):
        obs, info = test_env.reset()
        done = False
        total_reward = 0
        step = 0
        
        print(f"\nTest Episode: {ep+1}/{episodes}")
        
        while not done:
            discrete_state = get_discrete_state(obs)
            action = np.argmax(q_table[discrete_state])
            
            obs, reward, terminated, truncated, info = test_env.step(action)
            done = terminated or truncated
            
            total_reward += reward
            step += 1
            
            print(f"\rStep: {step} | Reward: {total_reward} | Position: {obs[0]:.2f} | Angle: {obs[2]:.2f}", end="")
            
            time.sleep(0.03)
        
        print(f"\nEpisode {ep+1} finished with reward: {total_reward}")
    
    test_env.close()
    print("="*50)

test_trained_model()