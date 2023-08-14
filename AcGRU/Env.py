'''
@Project ：AC 
@File    ：Env.py
@IDE     ：PyCharm 
@Author  ：chaoyi yang
@Date    ：2023/8/8 21:56 
'''

import numpy as np
import gym

class TaskSchedulingEnv(gym.Env):
    def __init__(self, adjacency_matrix):
        super(TaskSchedulingEnv, self).__init__()

        self.adjacency_matrix = adjacency_matrix
        self.num_nodes = adjacency_matrix.shape[0]
        self.reset()

        # Define action space and observation space
        self.action_space = gym.spaces.Discrete(self.num_nodes)
        self.observation_space = gym.spaces.Discrete(self.num_nodes)

    def reset(self):
        self.current_node = 0
        self.remaining_nodes = [node for node in range(self.num_nodes) if np.sum(self.adjacency_matrix[:, node]) == 0]
        self.reward = 0
        self.done = False

        return self.current_node

    def step(self, action):
        if action not in self.remaining_nodes:
            raise ValueError("Invalid action selected.")

        self.remaining_nodes.remove(action)

        # Update the adjacency matrix
        self.reward = 0
        self.adjacency_matrix[action, :] = 0
        self.adjacency_matrix[:, action] = 0

        self.current_node = action

        if len(self.remaining_nodes) == 0:
            self.done = True
            self.reward = np.sum(self.adjacency_matrix)

        return self.current_node, self.reward, self.done, {}

    def render(self, mode='human'):
        pass

    def close(self):
        pass


# Example adjacency matrix
adjacency_matrix = np.array([[0, 1, 1, 0],
                             [0, 0, 0, 1],
                             [0, 0, 0, 1],
                             [0, 0, 0, 0]])

# env = TaskSchedulingEnv(adjacency_matrix).step(0)
# print(env)

