"""
@Project ：AC 
@File    ：main.py
@IDE     ：PyCharm 
@Author  ：chaoyi yang
@Date    ：2023/8/9 22:30 
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from env.my_env import env

'''
    定义GRU模型作为ActorCritic的输入
    
    定义Actor网络
    策略
    '''


class Actor(nn.Module):
    def __init__(self, state_dim, hidden_dim, output_dim):
        super(Actor, self).__init__()
        self.gru = nn.GRU(state_dim, hidden_dim)
        # GRU output
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, hidden):
        output, hidden = self.gru(x.unsqueeze(0), hidden)
        action_probs = torch.softmax(self.output(output), dim=-1)
        return action_probs, hidden


'''
    定义Critic网络
    评价
    '''


class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(Critic, self).__init__()
        self.gru = nn.GRU(state_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, x, hidden):
        output, hidden = self.gru(x.unsqueeze(0), hidden)
        value = self.output(output)
        return value


'''
    定义Actor-Critic类
        
    接收GRU输出作为输入
    '''


# 定义Actor-Critic类，
class ActorCritic(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.actor = Actor(state_dim, hidden_dim, action_dim)
        self.critic = Critic(state_dim, hidden_dim)

    def forward(self, x, hidden):
        action_probs, hidden = self.actor(x, hidden)
        value = self.critic(x, hidden)
        return action_probs, value, hidden


'''
    定义 如何计算优势值 和 累计奖励 
    '''


def calculate_advantages(rewards, values, gamma=0.99, lamda=0.95):
    advantages = []
    gae = 0  # 广义优势估计

    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t + 1] - values[t]
        gae = delta + gamma * lamda * gae
        advantages.insert(0, gae)

    return advantages


def calculate_returns(rewards, gamma=0.99):
    returns = []
    G = 0  # 累积折扣奖励

    for t in reversed(range(len(rewards))):
        G = rewards[t] + gamma * G
        returns.insert(0, G)

    return returns


'''
    计算Actor和Critic的损失
    '''


def calculate_losses(log_probs, advantages, values, returns):
    # 策略梯度损失
    actor_loss = -torch.stack(log_probs).sum() * torch.stack(advantages).mean()

    # 均方误差损失
    critic_loss = nn.MSELoss()(torch.stack(values), torch.stack(returns))

    return actor_loss, critic_loss


'''
    定义训练过程
    '''


def train_actor_critic(env, model, num_epochs, learning_rate):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # criterion = nn.MSELoss()

    for i in range(10):
        with tqdm(total=int(num_epochs / 10), desc='进度 %d' % i) as pbar:
            for i_episode in range(int(num_epochs / 10)):
                for epoch in range(num_epochs):
                    state = env.reset()
                    hidden = None
                    log_probs = []
                    values = []
                    rewards = []

                    while not env.is_done():
                        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                        action_probs, value, hidden = model(state_tensor, hidden)
                        action_dist = torch.distributions.Categorical(action_probs)
                        action = action_dist.sample()

                        next_state, reward, done = env.step(action.item())

                        log_prob = action_dist.log_prob(action)
                        log_probs.append(log_prob)
                        values.append(value)
                        rewards.append(reward)

                        state = next_state
                    # 计算Advantage和Returns
                    advantages = calculate_advantages(rewards, values)
                    returns = calculate_returns(rewards)

                    # 计算Actor和Critic的损失
                    actor_loss, critic_loss = calculate_losses(log_probs, advantages, values, returns)

                    # 梯度更新
                    optimizer.zero_grad()
                    loss = actor_loss + critic_loss
                    loss.backward()
                    optimizer.step()
                pbar.update(1)


'''
    使用训练好的模型进行任务调度
    '''


def schedule_tasks(env, model):
    state = env.reset()
    hidden = None
    schedule = []

    while not env.is_done():
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action_probs, _, hidden = model(state_tensor, hidden)
        action = torch.argmax(action_probs).item()

        next_state, _, _ = env.step(action)
        schedule.append(action)

        state = next_state

    return schedule


if __name__ == "__main__":
    # 获取 初始 task 的 task_index 以及 adj_matrixer
    #
    adj_matrix = np.array([[0, 1, 1, 0, 0],
                           [0, 0, 0, 1, 0],
                           [0, 0, 0, 0, 1],
                           [0, 0, 0, 0, 1],
                           [0, 0, 0, 0, 0]
                           ])
    # 假设DAG是由节点和依赖关系构成的，依赖关系表示为一个邻接矩阵
    # 例如，dependencies[i][j] 表示任务 i 依赖任务 j
    dependencies = [[0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                    [0, 0, 0, 0]]

    # 状态维度、动作维度、隐藏状态维度
    state_dim = adj_matrix.shape[0] * adj_matrix.shape[0]  # 5
    print(state_dim)
    action_dim = 1  # 1
    hidden_dim = 128  # 128
    print(env.get_state_vec(adj_matrix, 0))

    # 初始化参数
    num_epochs = 100
    learning_rate = 0.001
    model = ActorCritic(state_dim, hidden_dim, action_dim)
    env = env

    # 训练Actor-Critic模型
    train_actor_critic(env, model, num_epochs, learning_rate)

    # 使用训练好的模型进行任务调度
    schedule = schedule_tasks(env, model)
    print("任务调度:", schedule)
