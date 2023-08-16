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
import pygraphviz as pgv
import re

# 读取 .gv 文件并解析为图形对象
file_path = "./env/data/meta_offloading_20/offload_random20_1/random.20.0.gv"
with open(file_path, "r") as f:
    gv_content = f.read()

# 提取节点和边的信息
node_pattern = r"(\d+) \[.*\]"
edge_pattern = r"(\d+) -> (\d+) .*"

nodes = re.findall(node_pattern, gv_content)
edges = re.findall(edge_pattern, gv_content)

# 确定节点的数量
num_nodes = max([int(node) for node in nodes])

# 初始化邻接矩阵
gen_adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
# 填充邻接矩阵
for src, dst in edges:
    src_idx = int(src) - 1
    dst_idx = int(dst) - 1
    gen_adjacency_matrix[src_idx, dst_idx] = 1


def zero_out(gen_adjacency_matrix):
    # in_degrees = np.sum(gen_adjacency_matrix, axis=0)
    out_degrees = np.sum(gen_adjacency_matrix, axis=1)
    zero_out_degrees_nodes = np.where(out_degrees == 0)[0]
    return zero_out_degrees_nodes


# print(zero_in_and_out(gen_adjacency_matrix))
zero_out_nodes = zero_out(gen_adjacency_matrix)
gen_adjacency_matrix[zero_out_nodes, zero_out_nodes] = 1
init_adjacency_matrix = gen_adjacency_matrix
# print(init_adjacency_matrix)
# 提取节点信息
node_pattern = r"(\d+) \[size=\"(\d+)\",.*\]"
nodes = re.findall(node_pattern, gv_content)

# 将提取的节点信息转换为字典，其中键为节点索引，值为节点的size（作为时延）
node_delay_map = {int(node[0]): int(node[1]) for node in nodes}

# print("节点时延映射:", node_delay_map)

'''
    定义GRU模型作为ActorCritic的输入
    
    定义Actor网络
    策略
    '''


class Actor(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Actor, self).__init__()
        self.gru = nn.GRU(state_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, action_dim)

    def forward(self, x, hidden):
        output, hidden = self.gru(x.unsqueeze(0), hidden)
        action_logits = self.output(output)

        action_probs = torch.softmax(action_logits, dim=-1)

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
    定义训练过程
    '''


def train_actor_critic(env, model, num_epochs, learning_rate):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()  # 负对数似然损失函数
    critic_criterion = nn.MSELoss()

    for epoch in range(num_epochs):
        state = env.reset()

        hidden = None
        log_probs = []
        values = []
        rewards = []
        schedule = []

        while True:
            # 获取可选动作

            action_space = env.get_action_space()
            action_space_size = env.num_nodes  # 动作空间长度
            available_actions = torch.zeros(action_space_size)  # 初始化为0
            available_actions[action_space] = 1.0  # 设置可选择的动作为1

            # 状态
            state_tensor = state.unsqueeze(0)
            # 模型输出 action_probs概率分布
            action_probs, value, hidden = model(state_tensor, hidden)

            masked_action_probs = action_probs * available_actions  # 将不可选择的动作的概率置为0
            action_probs_normalized = masked_action_probs / masked_action_probs.sum()  # 归一化概率

            # action_dist = torch.distributions.Categorical(action_probs)

            action = torch.argmax(action_probs_normalized).item()  # 选择概率最大的动作
            schedule.append(action)
            eps = 1e-8
            log_prob = torch.log(action_probs_normalized[0] + eps)  # 使用选择的动作的对数概率作为训练目标
            log_probs.append(log_prob)
            values.append(value)

            next_state, reward, done = env.step(action)
            rewards.append(float(reward))

            state = next_state
            # print(state)
            # print(schedule)


            if done:
                # 重置adjacency_matrix为原始输入的邻接矩阵
                # env.adjacency_matrix = np.array([[0, 1, 1, 0],
                #                                  [0, 0, 0, 1],
                #                                  [0, 0, 0, 1],
                #                                  [0, 0, 0, 0]])
                env.adjacency_matrix = np.copy(init_adjacency_matrix)

                print(f"done:{epoch + 1}/{num_epochs},Loss:{None},调度顺序为：{schedule} ")
                break

        # 梯度更新
        actor_loss = -torch.stack(log_probs).sum()  # 使用负对数似然作为损失
        critic_loss = critic_criterion(torch.tensor(rewards), torch.stack(values).squeeze())
        loss = actor_loss + critic_loss
        loss.backward()
        optimizer.step()


if __name__ == "__main__":
    # 初始化参数
    num_epochs = 1000
    learning_rate = 0.001

    # 假设DAG是由节点和依赖关系构成的，依赖关系表示为一个邻接矩阵
    # adjacency_matrix = np.array([[0, 1, 1, 0],
    #                              [0, 0, 0, 1],
    #                              [0, 0, 0, 1],
    #                              [0, 0, 0, 0]])
    adjacency_matrix = np.copy(init_adjacency_matrix)
    # print(adjacency_matrix)

    env = env(adjacency_matrix)
    action_space = env.get_action_space()
    model = ActorCritic(state_dim=env.num_nodes, hidden_dim=32, action_dim=env.num_nodes)
    # 训练Actor-Critic模型
    train_actor_critic(env, model, num_epochs, learning_rate)

    # 使用训练好的模型进行任务调度
    state = env.reset()
    hidden = None
    schedule = []

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
