"""
@Project ：AC
@File    ：main.py
@IDE     ：PyCharm
@Author  ：chaoyi yang
@Date    ：2023/8/9 22:30
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import random

from env.offloading_my_env import OffloadingEnvironment
from env.offloading_my_env import TaskGraph

'''
    定义GRU模型作为ActorCritic的输入
    
    定义Actor网络
    策略
    '''


class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, unload_dim):
        super(PolicyNet, self).__init__()
        self.gru = nn.GRU(state_dim, hidden_dim)
        self.unload_output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, unload_dim)
        )

    def forward(self, state):
        output, hidden = self.gru(state)
        probs = torch.softmax(self.unload_output(output), dim=1)
        return probs


class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        value = self.fc2(x)
        return value


'''
    定义Actor-Critic类  
    接收GRU输出作为输入
    '''


class ActorCritic:
    def __init__(self, state_dim, hidden_dim, unload_dim,
                 actor_lr, critic_lr, gamma, device):
        # 策略网络
        self.actor = PolicyNet(state_dim, hidden_dim, unload_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)  # 价值网络
        # 策略网络优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)  # 价值网络优化器
        self.gamma = gamma
        self.device = device

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.actor(state)
        # print('probs:', probs)

        epsilon = 0.12
        if random.random() < epsilon:
            # 以 epsilon 的概率随机选择一个动作
            action = random.randint(0, 1)
        else:
            # 以 1 - epsilon 的概率选择具有最高概率的动作
            action = torch.argmax(probs).item()

        # action = torch.argmax(probs).item()
        # action_dist = torch.distributions.Categorical(probs)
        # action = torch.distributions.Categorical(probs).sample().item()
        # action = 1
        return action

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)
        # 时序差分目标
        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_delta = td_target - self.critic(states)  # 时序差分误差
        log_probs = torch.log(self.actor(states).gather(1, actions))
        actor_loss = torch.mean(-log_probs * td_delta.detach())
        # 均方误差损失函数
        critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        actor_loss.backward()  # 计算策略网络的梯度
        critic_loss.backward()  # 计算价值网络的梯度
        self.actor_optimizer.step()  # 更新策略网络的参数
        self.critic_optimizer.step()  # 更新价值网络的参数


'''
    定义训练过程
    '''


def train_on_policy_agent(schedule_list, env, agent, num_episodes):
    return_list = []
    time_list = []
    for i in range(20):
        with tqdm(total=int(num_episodes / 20), desc="Iteration %d" % (i+1)) as pbar:
            for i_episode in range(int(num_episodes / 20)):
                episode_return = 0
                transition_dict = {'states': [],
                                   'hiddens': [],
                                   'actions': [],
                                   'next_states': [],
                                   'rewards': [],
                                   'dones': [],
                                   'time': []}

                index = 0
                done = False
                env.reset()
                while index < len(schedule_list) - 1:
                    task = schedule_list[index]
                    next_task = schedule_list[index + 1]
                    state = env.offloading_state_list()[task]  # 卸载状态

                    state[17] = env.local_available_time
                    state[18] = env.edge_available_time

                    if not transition_dict['actions']:
                        pass
                    else:
                        last_action = transition_dict['actions'][-1]
                        state[18 + task] = last_action
                    # print(state)

                    action = agent.take_action(state)
                    reward, current_time = env.offloading_step(action, task)

                    next_state = env.offloading_state_list()[next_task]
                    next_state[17] = env.local_available_time
                    next_state[18] = env.edge_available_time
                    next_state[18 + task] = action

                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done)
                    episode_return += reward
                    index += 1
                    if next_task == -1:
                        done = True
                time_list.append(current_time)
                # print(f'episode:{i_episode},actions:{transition_dict["actions"]}')
                # print(f'episode:{i_episode},local:edge-->{env.local_available_time}:{env.edge_available_time}')
                # print(f'episode:{i_episode},time:{current_time}')
                return_list.append(episode_return)
                agent.update(transition_dict)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes / 20 * i + i_episode + 1),
                                      'return': '%.3f' % np.mean(return_list[-10]),
                                      'avg_time': '%.3f' % np.mean(time_list[-10])
                                      })
                pbar.update(1)
    return return_list


'''
    平均化
    '''


def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0))
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size - 1, 2)
    begin = np.cumsum(a[:window_size - 1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))


if __name__ == "__main__":
    # 参数
    actor_lr = 1e-5
    critic_lr = 1e-6
    num_episodes = 2000
    gamma = 0.98
    device = "cpu"
    # 模型参数
    state_dim = 39
    hidden_dim = 128
    unload_dim = 2

    # GRU+AC建模
    agent = ActorCritic(state_dim, hidden_dim, unload_dim,
                        actor_lr, critic_lr, gamma, device)
    # 调度顺序
    dag_schedule_list = []
    a = [3, 5, 9, 2, 1, 10, 4, 8, 11, 7, 12, 14, 18, 15, 19, 6, 13, 17, 16, 20, -1]
    # b = [4, 5, 3, 7, 2, 1, 6, 8, 12, 11, 10, 9, 15, 19, 16, 13, 14, 20, 18, 17, -1]
    # c = [1, 2, 3, 9, 5, 7, 4, 10, 12, 8, 13, 19, 6, 14, 15, 20, 11, 17, 16, 18, -1]
    dag_schedule_list.append(a)
    # dag_schedule_list.append(b)
    # dag_schedule_list.append(c)
    # train on policy
    for i in range(1):
        # DAG图文件地址
        file_path = "./env/data/meta_offloading_20/offload_random20_1/random.20.{}.gv".format(i)
        # 图 结构
        task_graph = TaskGraph(file_path)
        # 卸载环境
        env = OffloadingEnvironment(mec_process_capable=(10.0 * 1024 * 1024),
                                    mobile_process_capable=(1.0 * 1024 * 1024),
                                    bandwidth_up=7.0,
                                    bandwidth_dl=7.0,
                                    graph_file_paths=file_path)
        # 调度list
        schedule_list = dag_schedule_list[i]
        return_list = train_on_policy_agent(schedule_list, env, agent, num_episodes)
        episodes_list = list(range(len(return_list)))
        mv_return = moving_average(return_list, 9)
        plt.plot(episodes_list, mv_return)
        plt.xlabel('Episodes')
        plt.ylabel('Returns')
        plt.title('Smooth Actor-Critic on DAG{}'.format(i + 1))
        plt.show()
