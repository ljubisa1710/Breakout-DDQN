import gymnasium as gym  # Importing the environment library
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter  # For logging and visualizing in TensorBoard
import os
import shutil
import random

class DuelingCNNDQN(nn.Module):
    def __init__(self, n_actions):
        super(DuelingCNNDQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        # Value stream with additional hidden layers
        self.value_stream = nn.Sequential(
            nn.Linear(22528, 512),
            nn.ReLU(),
            nn.Linear(512, 128),  # Additional hidden layer
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # Value stream with additional hidden layers
        self.advantage_stream = nn.Sequential(
            nn.Linear(22528, 512),
            nn.ReLU(),
            nn.Linear(512, 128),  # Additional hidden layer
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        return value + (advantage - advantage.mean())

# Define the DQN Agent
class DoubleDuelingDQNAgent:
    def __init__(self, state_dim, action_dim, learning_rate=0.0005, gamma=0.99, 
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay_steps=1000, target_network_update_frequency=20, max_memory_size=100_000, batch_size=64):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.criterion = nn.HuberLoss()  # Loss function
        self.gamma = gamma  # Discount factor for future rewards
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_decay_strategy = self.linear_decay
        self.epsilon_decay_steps = epsilon_decay_steps
        self.epsilon_min = epsilon_min  # Minimum exploration rate
        self.memory = []  # Memory to store experiences for replay
        self.max_memory_size = max_memory_size
        self.target_network_update_frequency = target_network_update_frequency
        self.batch_size = batch_size

        self.q_net = DuelingCNNDQN(self.action_dim).to(device)
        self.target_q_net = DuelingCNNDQN(self.action_dim).to(device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())

        # Optimization and learning rate scheduling
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=learning_rate)

    def update_epsilon(self):
        self.epsilon = self.epsilon_decay_strategy()

    # Function for the agent to take actions
    def select_action(self, states):
        actions = []
        for state in states:
            if np.random.rand() <= self.epsilon:
                actions.append(random.choice(range(action_dim)))  # Random action (exploration)
            else:
                state_tensor = torch.tensor(state.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
                q_values = self.q_net(state_tensor)
                actions.append(torch.argmax(q_values).item())
        return actions  # Best action based on current policy (exploitation)

    # Function to store experiences
    def remember(self, states, actions, rewards, next_states, dones):
        # Ensure that all variables are lists or arrays
        if not isinstance(states, list):
            states = [states]
        if not isinstance(actions, list):
            actions = [actions]
        if not isinstance(rewards, list):
            rewards = [rewards]
        if not isinstance(next_states, list):
            next_states = [next_states]
        if not isinstance(dones, list):
            dones = [dones]

        for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
            if len(self.memory) == self.max_memory_size:  # If memory is full
                self.memory.pop(0)  # Remove the oldest experience
            self.memory.append((state, action, reward, next_state, done))


    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)

        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                next_state_tensor = torch.tensor(next_state.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
                next_state_actions = self.q_net(next_state_tensor)
                best_action = torch.argmax(next_state_actions).item()
                
                next_state_values = self.target_q_net(next_state_tensor)
                target += self.gamma * next_state_values[0][best_action].item()

            state_tensor = torch.tensor(state.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
            current_q = self.q_net(state_tensor)[0][action]
            loss = self.criterion(current_q.unsqueeze(0), torch.tensor([target]).float().to(device))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return loss
    
    def linear_decay(self):
        decay = (self.epsilon - self.epsilon_min) / self.epsilon_decay_steps
        new_epsilon = max(self.epsilon - decay, self.epsilon_min)
        return new_epsilon

# Function to load a pre-trained model
def load_pretrained_model(agent, model_path=None):
    if model_path is not None and os.path.exists(model_path):
        agent.q_net.load_state_dict(torch.load(model_path))
        print(f"Model loaded from {model_path}")
        return True
    elif model_path is not None:
        print(f"Model path {model_path} does not exist. Starting from scratch.")
        return False
    else:
        print("No model path provided. Starting from scratch.")
        return False

def watch_agent_play(env, agent, num_envs, env_index=0, episodes=5):
    agent.q_net.eval()
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action = agent.select_action(obs[env_index])  # Choose an action based on a specific environment observation
            actions = [0] * num_envs
            actions[env_index] = action[env_index] 
            _, _, terminated, truncated, _ = env.step(actions)
            done = terminated[env_index] or truncated[env_index]

def save_file(save_name):
    # Define the path of the file to be copied
    source_file_path = "breakout.py"

    # Define the directory where the file should be copied
    destination_directory = f"saved_models/{save_name}"

    # Check if the destination directory exists, if not create it
    if not os.path.exists(destination_directory):
        os.makedirs(destination_directory)

    # Define the path where the file will be copied
    destination_file_path = os.path.join(destination_directory, f"breakout_{save_name}.py")

    # Copy the file
    shutil.copy2(source_file_path, destination_file_path)

    print(f"File copied to {destination_file_path}")

def save_model(save_name, episode):
    if not os.path.exists(f"saved_models/{save_name}"):
    
        os.makedirs(f"saved_models/{save_name}")

    model_save_path = f"saved_models/{save_name}/model_episode_{episode}.pth"
    torch.save(agent.q_net.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

def log(episode, episode_reward):
    # Logging to Tensorboard
    writer.add_scalar('Episode Reward', np.mean(episode_reward), episode)
 
def train_agent(env, agent, seed=None, loss_update_frequency=100, save_frequency=10, episode=0):
    try:
        done = np.zeros(env.num_envs)
        episode_reward = np.zeros(env.num_envs)
        stat_episode_reward = np.zeros(env.num_envs)
        best_rewards = np.zeros(env.num_envs)
        obs, _ = env.reset(seed=seed)

        while True:
            episode += 1

            actions = agent.select_action(obs)
            next_obs, rewards, terminated, truncated, _ = env.step(actions)
            

            for i in range(env.num_envs):
                agent.remember(obs[i], actions[i], rewards[i], next_obs[i], terminated[i] or truncated[i])
                done[i] = terminated[i] or truncated[i]

                if (done[i] == 1):
                    stat_episode_reward[i] = 0

            obs = next_obs
            episode_reward += rewards
            stat_episode_reward = episode_reward

            for i in range(env.num_envs):
                if (stat_episode_reward[i] > best_rewards[i]):
                    best_rewards[i] = stat_episode_reward[i]

            # Update the agent every loss_update_frequency frames
            if episode % loss_update_frequency == 0:
                print(f"Collected Reward:   {stat_episode_reward}")
                print(f"Best Episodes:      {best_rewards}")
                print(f"Current Exploration rate: {agent.epsilon}")
                print(f"Calculating Loss at timestep {episode}")
                log(episode, best_rewards)
                for _ in range(env.num_envs):
                    loss = agent.replay()
                    if loss is not None:
                        np_loss = torch.Tensor.numpy(loss, force=True)
                        print("\t" + str(np_loss)) 

            # Update the target network every few frames
            if episode % agent.target_network_update_frequency == 0:
                agent.target_q_net.load_state_dict(agent.q_net.state_dict())
                print(f"Target network updated at timestep {episode}")

            agent.update_epsilon()

            # Save the model every save_frequency timesteps
            if episode % save_frequency == 0:
                save_model(save_name, episode)

    except Exception as e:
        print(f"An error occurred: {e}")
        pass

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Using GPU if available

    save_name = "v1.1"

    # TensorBoard setup for logging
    writer = SummaryWriter(log_dir=f"runs/{save_name}")

    save_file(save_name)

    seed = random.randint(0, 1000)
    n_envs = 3

    # Environment setup
    env = gym.vector.make("ALE/Breakout-v5", num_envs=n_envs)
    state_dim = env.single_observation_space.shape[0]
    action_dim = env.single_action_space.n
    agent = DoubleDuelingDQNAgent(state_dim, action_dim, learning_rate=0.0005, epsilon_decay_steps=100_000, target_network_update_frequency=10_000, batch_size=32)
    # load_pretrained_model(agent, "saved_models/v1.8_Assault/model_episode_4.pth")

    # 36_000 timesteps === 10.0 mins gameplay at 60 fps
    train_agent(env, agent, seed, save_frequency=10_000, loss_update_frequency=5)

    writer.close()
    env.close()

    # watch_env = gym.vector.make("ALE/Assault-v5", num_envs=n_envs, render_mode="human")
    # state_dim = watch_env.single_observation_space.shape[0]
    # action_dim = watch_env.single_action_space.n
    # agent = DoubleDuelingDQNAgent(state_dim, action_dim)
    # # load_pretrained_model(agent, "saved_models/v1.7.3_DDQN_N_Envs/model_episode_5.pth")
    # watch_agent_play(watch_env, agent, num_envs=n_envs)
    # watch_env.close()