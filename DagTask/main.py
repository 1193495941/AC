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
from env.my_env import schedule_env
from env.my_env import unload_env
from env.my_env import taskGraph
import math
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 读取 .gv 文件并解析为图形对象
file_path = "./env/data/meta_offloading_20/offload_random20_1/random.20.0.gv"

# 生成初始化并预处理的邻接矩阵
init_adjacency_matrix = taskGraph(file_path).gen_adjacency_matrix()

# 将提取的节点信息转换为字典，其中键为节点索引，值为节点的size(节点的data_size)
# node_delay_map = taskGraph(file_path).get_data_size()

# 获取最坏时间
# bt = taskGraph(file_path).get_bad_time()  # 1271902208

'''
    定义GRU模型作为ActorCritic的输入
    
    定义Actor网络
    策略
    '''


class Actor(nn.Module):
    def __init__(self, state1_dim, state2_dim, hidden_dim, action_dim, unload_dim=2):
        # state_dim = action_dim = num_nodes; unload_dim = 2
        super(Actor, self).__init__()
        self.gru1 = nn.GRU(state1_dim, hidden_dim)
        self.gru2 = nn.GRU(state2_dim, hidden_dim)
        self.schedule_output = nn.Linear(hidden_dim, action_dim)
        self.unload_output = nn.Linear(hidden_dim, unload_dim)  # 新增的输出层用于卸载位置

    def forward(self, x1, x2, hidden1, hidden2):
        output1, hidden1 = self.gru1(x1.unsqueeze(0), hidden1)
        output2, hidden2 = self.gru2(x2.unsqueeze(0), hidden2)  # 使用第二个 GRU 层

        action_logits = self.schedule_output(output1)
        unload_logits = self.unload_output(output2)  # 获取卸载位置的 unload_logits

        action_probs = torch.softmax(action_logits, dim=-1)  # 调度节点的概率分布
        unload_probs = torch.softmax(unload_logits, dim=-1)  # 卸载位置的概率分布

        return action_probs, unload_probs, hidden1, hidden2


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
    def __init__(self, state1_dim, state2_dim, hidden_dim, action_dim, unload_dim=2):
        super(ActorCritic, self).__init__()
        self.actor = Actor(state1_dim, state2_dim, hidden_dim, action_dim, unload_dim)
        self.critic1 = Critic(state1_dim, hidden_dim)
        self.critic2 = Critic(state2_dim, hidden_dim)

    def forward(self, x, y, hidden1, hidden2):
        action_probs, unload, hidden1, hidden2 = self.actor(x, y, hidden1, hidden2)
        value1 = self.critic1(x, hidden1)
        value2 = self.critic2(y, hidden2)
        return action_probs, unload, value1, value2, hidden1, hidden2


'''
    定义训练过程
    '''


def train_actor_critic(s_env, u_env, model, num_epochs, learning_rate):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()  # 负对数似然损失函数
    critic_criterion = nn.MSELoss()

    epochs = []
    epoch_rewards = []
    for epoch in range(num_epochs):
        state1 = s_env.reset()
        state2 = u_env.reset()
        hidden1 = None
        hidden2 = None
        log_probs1 = []
        log_probs2 = []
        values1 = []
        values2 = []
        rewards = []

        schedule = []  # 节点调度顺序
        unload = []  # 卸载位置顺序
        local_schedule = []  # 本地端节点顺序
        edge_schedule = []  # 边缘端节点顺序

        while True:
            # 调度环境
            action_space = s_env.get_action_space()  # 获取可选动作
            action_space_size = s_env.num_nodes  # 动作空间长度
            available_actions = torch.zeros(action_space_size)  # 初始化为0
            available_actions[action_space] = 1.0  # 设置可选择的动作为1

            # 卸载环境
            unload_action_space = u_env.get_action_space()

            # 状态1
            state_tensor1 = state1.unsqueeze(0)  # 所有节点是否被调选作为状态
            # 状态2
            state_tensor2 = state2.unsqueeze(0)  # 卸载状态

            # 模型输出 action_probs概率分布 and unload_action概率分布
            schedule_action_probs, unload_action_probs, value1, value2, hidden1, hidden2 = model(state_tensor1,
                                                                                                 state_tensor2,
                                                                                                 hidden1,
                                                                                                 hidden2)

            masked_action_probs = schedule_action_probs * available_actions  # 将不可选择的动作的概率置为0
            action_probs_normalized = masked_action_probs / masked_action_probs.sum()  # 归一化概率
            # action_dist = torch.distributions.Categorical(action_probs)
            # 调度动作
            action1 = torch.argmax(action_probs_normalized).item()  # 选择概率最大的动作
            schedule.append(action1)
            eps = 1e-8
            log_prob1 = torch.log(action_probs_normalized[0] + eps)  # 使用选择的动作的对数概率作为训练目标
            log_probs1.append(log_prob1)


            # （在调度前获取节点调度情况） 未调度的节点默认在本地运行
            flipped_nodes_done1 = [1 if item == 0 else 0 for item in s_env.nodes_done]
            remain_size1 = 0
            remain_i = 0
            for i in flipped_nodes_done1:
                y = taskGraph(file_path).get_data_size(remain_i) * i
                remain_size1 = remain_size1 + y
                remain_time1 = u_env.locally_execution_cost(remain_size1)
                remain_i = remain_i +1


            next_state1, reward1, done = s_env.step(action1)

            # （在调度前获取节点调度情况） 未调度的节点默认在本地运行
            flipped_nodes_done2 = [1 if item == 0 else 0 for item in s_env.nodes_done]
            remain_size2 = 0
            remain_j = 0
            for j in flipped_nodes_done2:
                y = taskGraph(file_path).get_data_size(remain_j) * j
                remain_size2 = remain_size2 + y
                remain_time2 = u_env.locally_execution_cost(remain_size2)
                remain_j = remain_j + 1

            # rewards.append(float(reward1))

            # 卸载动作
            action2 = torch.argmax(unload_action_probs)
            unload.append(action2)
            log_prob2 = torch.log(unload_action_probs[0] + eps)  # 使用选择的动作的对数概率作为训练目标
            log_probs2.append(log_prob2)

            values1.append(value1)
            values2.append(value2)
            # values = values1 + values2
            values = [(x + y) / 2 for x, y in zip(values1, values2)]

            # 状态的转移
            state1 = next_state1

            # print(state)
            # print(schedule)

            if done:
                # 提交reward
                # reward2 = u_env.step(file_path, action2, action1)
                bad_time, current_time = u_env.step(file_path, action2, action1)
                reward2 = bad_time - (remain_time2 + current_time)  # 调度前的总时间 - 调度后的总时间 (正数)
                reward = (reward1 + reward2)
                rewards.append(float(reward))
                epoch_rewards.append(reward)
                schedule_time = u_env.current_FT

                # print('rewards',rewards)
                # print('reward1',reward1)
                # print('reward2',reward2)


                # 重置adjacency_matrix为原始输入的邻接矩阵
                # env.adjacency_matrix = np.array([[0, 1, 1, 0],
                #                                  [0, 0, 0, 1],
                #                                  [0, 0, 0, 1],
                #                                  [0, 0, 0, 0]])
                s_env.adjacency_matrix = np.copy(init_adjacency_matrix)
                epochs.append(epoch)
                print(f"done:{epoch + 1}/{num_epochs},reward:{rewards} ")
                print(f"调度时间:{schedule_time},调度顺序为：{schedule}")
                break
            else:
                # 提交reward
                # reward2 = 0
                bad_time, current_time = u_env.step(file_path, action2, action1)
                # print('badtime', bad_time)
                # print('current_time', current_time)
                # print('remain_time1',remain_time2)
                reward2 = bad_time - (remain_time2 + current_time)  # 调度前的总时间 - 调度后的总时间 (正数)
                reward = (reward1 + reward2)
                rewards.append(float(reward))

        # 梯度更新
        actor_loss1 = -torch.stack(log_probs1).sum()  # 使用负对数似然作为损失
        actor_loss2 = -torch.stack(log_probs2).sum()  # 使用负对数似然作为损失
        actor_loss = actor_loss1 + actor_loss2
        # critic_loss = critic_criterion(torch.tensor(rewards), torch.stack(values).squeeze())
        critic_loss = critic_criterion(torch.tensor(rewards), torch.stack(values).squeeze())
        loss = actor_loss + critic_loss
        # 清除之前的梯度
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return epochs, epoch_rewards


if __name__ == "__main__":
    # 初始化参数
    num_epochs = 2000
    learning_rate = 0.0001
    # learning_rate = 0.00001

    # 假设DAG是由节点和依赖关系构成的，依赖关系表示为一个邻接矩阵
    # adjacency_matrix = np.array([[0, 1, 1, 0],
    #                              [0, 0, 0, 1],
    #                              [0, 0, 0, 1],
    #                              [0, 0, 0, 0]])
    adjacency_matrix = np.copy(init_adjacency_matrix)

    s_env = schedule_env(adjacency_matrix)  # 调度环境
    u_env = unload_env(edge_process_capable=(10.0 * 1024 * 1024),
                       local_process_capable=(1.0 * 1024 * 1024), bandwidth_up=7.0, bandwidth_dl=7.0)  # 卸载环境
    action_space = s_env.get_action_space()
    model = ActorCritic(state1_dim=s_env.num_nodes, state2_dim=2, hidden_dim=32, action_dim=s_env.num_nodes,
                        unload_dim=2)
    # 训练Actor-Critic模型
    epochs, epoch_rewards = train_actor_critic(s_env, u_env, model, num_epochs, learning_rate)

    '''
            epochs和reward 图
            '''

    # 使用训练好的模型进行任务调度
    # state = env.reset()
    # hidden = None
    # schedule = []

    # while True:
    #     action_space = env.get_action_space()
    #     state_tensor = state.unsqueeze(0)
    #     action_probs, _, hidden = model(state_tensor, hidden)
    #
    #     available_actions = torch.tensor([action_space], dtype=torch.float32)
    #     masked_action_probs = action_probs * available_actions
    #
    #     action = torch.argmax(masked_action_probs).item()
    #
    #     next_state, _, done = env.step(action)
    #     schedule.append(action)
    #
    #     # 将已调度节点的行置零
    #     adjacency_matrix[action, :] = 0
    #     # 重置adjacency_matrix为原始输入的邻接矩阵
    #     if done:
    #         adjacency_matrix = np.array([[0, 1, 1, 0],
    #                                      [0, 0, 0, 1],
    #                                      [0, 0, 0, 1],
    #                                      [0, 0, 0, 0]])
    #
    #     state = next_state
    #     break
    #
    # print("Task schedule:", schedule)
