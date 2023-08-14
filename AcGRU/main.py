'''
@Project ：AC 
@File    ：main.py
@IDE     ：PyCharm 
@Author  ：chaoyi yang
@Date    ：2023/8/8 21:56 
'''

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gym


# Define Actor and Critic networks
class Actor(nn.Module):
    def __init__(self, input_size, output_size):
        super(Actor, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        return torch.softmax(self.fc(x), dim=-1)


class Critic(nn.Module):
    def __init__(self, input_size):
        super(Critic, self).__init__()
        self.fc = nn.Linear(input_size, 1)

    def forward(self, x):
        return self.fc(x)


# Define the AC Agent
class ACAgent:
    def __init__(self, input_size, output_size, lr_actor=0.001, lr_critic=0.001, gamma=0.99):
        self.actor = Actor(input_size, output_size)
        self.critic = Critic(input_size)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.gamma = gamma

    def select_action(self, state):
        state_tensor = torch.tensor([state]).float()  # Convert scalar state to a tensor
        probabilities = self.actor(state_tensor)
        action = np.random.choice(len(probabilities), p=probabilities.detach().numpy())
        return action

    def update(self, state, action, reward, next_state, done):
        state = torch.tensor(state).float()
        next_state = torch.tensor(next_state).float()
        action_probs = self.actor(state)
        action_log_prob = torch.log(action_probs[action])
        value = self.critic(state)
        next_value = self.critic(next_state) if not done else 0

        # Calculate advantage and update critic
        advantage = reward + self.gamma * next_value - value
        critic_loss = advantage ** 2
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update actor
        actor_loss = -action_log_prob * advantage
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()


# Define the Task Scheduling Environment
class TaskSchedulingEnv(gym.Env):
    def __init__(self, adjacency_matrix):
        super(TaskSchedulingEnv, self).__init__()

        self.adjacency_matrix = adjacency_matrix
        self.num_nodes = adjacency_matrix.shape[0]
        self.all_nodes = list(range(self.num_nodes))
        self.reset()

        self.action_space = gym.spaces.Discrete(self.num_nodes)
        self.observation_space = gym.spaces.Discrete(self.num_nodes)

    def reset(self):
        self.adj_matrix = self.adjacency_matrix.copy()
        self.remaining_nodes = self.all_nodes.copy()
        self.current_node = 0
        self.reward = 0
        self.done = False
        return self.current_node

    def step(self, action):
        if action not in self.remaining_nodes:
            raise ValueError("Invalid action selected.")

        self.remaining_nodes.remove(action)

        # Update adjacency matrix
        self.reward += np.sum(self.adj_matrix[action, :])
        self.adj_matrix[action, :] = 0
        self.adj_matrix[:, action] = 0

        self.current_node = action

        if len(self.remaining_nodes) == 0:
            self.done = True
            self.reward += np.sum(self.adj_matrix)

        return self.current_node, self.reward, self.done, {}

    def render(self, mode='human'):
        pass

    def close(self):
        pass


# Main training loop
def main():

    adjacency_matrix = np.array([[0, 1, 1, 0],
                                 [0, 0, 0, 1],
                                 [0, 0, 0, 1],
                                 [0, 0, 0, 0]])
    env = TaskSchedulingEnv(adjacency_matrix)
    agent = ACAgent(input_size=env.num_nodes, output_size=env.num_nodes)

    num_episodes = 1000
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            agent.update(state, action, reward, next_state, done)
            state = next_state

        print(f"Episode {episode + 1}, Total Reward: {total_reward}")


if __name__ == "__main__":
    main()
