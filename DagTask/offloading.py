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
import numpy as np

from env.offloading_my_env import OffloadingEnvironment
from env.offloading_my_env import TaskGraph

import random

'''
    定义GRU模型作为ActorCritic的输入
    
    定义Actor网络
    策略
    '''


class Actor(nn.Module):
    def __init__(self, state_dim, hidden_dim, unload_dim=2):
        # state_dim = action_dim = num_nodes; unload_dim = 2
        super(Actor, self).__init__()
        self.gru = nn.GRU(state_dim, hidden_dim)
        # self.unload_output = nn.Linear(hidden_dim, unload_dim)
        # 新增的输出层用于卸载位置
        self.unload_output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, unload_dim)
        )

    def forward(self, state, hidden):
        output, hidden = self.gru(state.unsqueeze(0), hidden)
        unload_logits = self.unload_output(output)  # 获取卸载位置的 unload_logits
        unload_probs = torch.softmax(unload_logits, dim=-1)  # 卸载位置的概率分布
        return unload_probs, hidden


'''
    定义Critic网络
    评价
    '''


class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(Critic, self).__init__()
        self.gru = nn.GRU(state_dim, hidden_dim)
        # self.output = nn.Linear(hidden_dim, 1)
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, hidden):
        output, hidden = self.gru(state.unsqueeze(0), hidden)
        value = self.output(hidden)
        return value


'''
    定义Actor-Critic类  
    接收GRU输出作为输入
    '''


# 定义Actor-Critic类，
class ActorCritic(nn.Module):
    def __init__(self, state_dim, hidden_dim, unload_dim=2):
        super(ActorCritic, self).__init__()
        self.actor = Actor(state_dim, hidden_dim, unload_dim)
        self.critic = Critic(state_dim, hidden_dim)

    def forward(self, state, hidden):
        action, next_hidden = self.actor(state, hidden)
        value = self.critic(state, hidden)
        return action, value, next_hidden


'''
    定义训练过程
    '''


def train_actor_critic(u_env, ac_model, num_epochs, lr):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    critic_criterion = nn.MSELoss()
    epochs = []
    epoch_rewards = []
    for epoch in range(num_epochs):
        u_env.reset()
        hidden = None
        log_probs = []
        values = []
        rewards = []
        unload = []  # 卸载位置顺序

        # 调度顺序
        schedule_list = [2, 4, 8, 1, 0, 9, 3, 7, 10, 6, 11, 13, 17, 14, 18, 5, 12, 16, 15, 19]
        all_local_time = 0.0
        for i in schedule_list:
            task = i + 1
            task_size = u_env.locally_execution_cost(task_graph.get_node_size(task))
            all_local_time = all_local_time + task_size

        for i in schedule_list:
            task = i + 1
            # 状态
            state = u_env.offloading_state_list()[task]  # 卸载状态
            # print(state)
            state_tensor = torch.tensor(state, dtype=torch.float32)
            # 模型输出 unload_action 概率分布
            unload_action_probs, value, hidden = ac_model(state_tensor, hidden)

            # （在调度前获取节点调度情况） 未调度的节点默认在本地运行
            pre_offloading = all_local_time
            # （在调度前获取节点调度情况） 未调度的节点默认在本地运行
            after_offloading = all_local_time - u_env.locally_execution_cost(task_graph.get_node_size(task))
            all_local_time = after_offloading  # 更新剩余本地调度时间
            # 卸载动作  根据ϵ-greedy算法选择动作
            epsilon = 0.1  # 选择随机动作的概率
            if random.random() < epsilon:
                # 以ϵ的概率选择随机动作
                offloading_action = torch.randint(0, unload_action_probs.shape[-1],
                                                  (1,)).item()  # 替换num_possible_actions为你的动作空间大小
            else:
                # 以1-ϵ的概率选择基于模型输出的动作
                offloading_action = torch.argmax(unload_action_probs).item()
                # offloading_action = torch.randint(0, unload_action_probs.shape[-1], (1,)).item()

            unload.append(offloading_action)
            values.append(value)

            reward, schedule_time = u_env.offloading_step(offloading_action, task, pre_offloading,
                                                          after_offloading)

            # 优势值
            advantage = reward - float(value)
            actor_loss = -torch.log(unload_action_probs[0, offloading_action]) * advantage
            critic_loss = critic_criterion(value, torch.tensor([[[reward]]]))
            loss = actor_loss + critic_loss
            optimizer.zero_grad()
            # loss.backward(retain_graph=True)
            torch.autograd.grad(loss, ac_model.parameters(), retain_graph=True)
            optimizer.step()
            # optimizer.zero_grad()
            print(f'epoch:{epoch}    reward:{round(reward, 4)}    loss:{loss}')
        print(f'time:{round(schedule_time, 4)}')
        print(f'unload:{unload}')
        # print(f'dag_time:{dag_time}')


if __name__ == "__main__":
    num_epochs = 1000
    learning_rate = 0.0001
    file_path = "./env/data/meta_offloading_20/offload_random20_1/random.20.0.gv"

    # 图 结构
    task_graph = TaskGraph(file_path)
    # 卸载环境
    env = OffloadingEnvironment(mec_process_capable=(10 * 1024 * 1024),
                                mobile_process_capable=(1.0 * 1024 * 1024),
                                bandwidth_up=14.0,
                                bandwidth_dl=14.0,
                                graph_file_paths=file_path)
    # GRU+AC模型
    model = ActorCritic(state_dim=17, hidden_dim=512, unload_dim=2)
    # 训练模型
    train_actor_critic(env, model, num_epochs, learning_rate)
