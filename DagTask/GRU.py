'''
@Project ：AC 
@File    ：GPolicy.py
@IDE     ：PyCharm 
@Author  ：chaoyi yang
@Date    ：2023/8/8 23:08 
'''
import random

import numpy as np
import torch.nn as nn
import torch
from torch.nn.init import xavier_normal_, xavier_uniform_
import torch.nn.functional as F
import torch.optim as optim


# Actor-Critic Network
class ActorCritic(nn.Module):
    def __init__(self, input_size, output_size):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.actor = nn.Linear(128, output_size)
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        policy = F.softmax(self.actor(x), dim=-1)
        value = self.critic(x)
        return policy, value


class gate(nn.Module):
    def __init__(self, input, output, activation='sigmoid'):
        super(gate, self).__init__()
        self.l_x = nn.Linear(input, output, bias=False)
        self.l_h = nn.Linear(output, output, bias=False)
        xavier_uniform_(self.l_x.weight.data)
        xavier_uniform_(self.l_h.weight.data)
        if activation == 'sigmoid':
            self.activation = torch.sigmoid
        elif activation == 'tanh':

            self.activation = torch.tanh

        elif activation == 'relu':
            self.activation = torch.nn.functional.relu

    def forward(self, x, h):
        res1 = self.l_x(x)
        res2 = self.l_h(h)

        res = self.activation(res1 + res2)
        return res


class Policy:
    def __init__(self, input=10, hidden=10, output=10):
        self.history = []
        self.hidden = hidden
        self.hidden_vec = torch.zeros(self.hidden)

        self.resgate = gate(input, hidden, 'sigmoid')
        self.zgate = gate(input, hidden, 'sigmoid')

        self.enco = gate(input, hidden, 'tanh')

        self.policy_decoder = nn.Linear(hidden + input, output, bias=True)
        self.n_action = 16
        self.actor_critic = ActorCritic(output, self.n_action)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=0.001)

    def calcuHistory(self):
        if len(self.history) > 0:
            for i in self.history:
                e_h, r_h, target_e_h = i[0], i[1], i[2]
                # tt = self.env.get_state_vec(e_h, r_h, target_e_h)
                tt = torch.tensor(1000)

                r = self.resgate(tt, self.hidden_vec)
                z = self.zgate(tt, self.hidden_vec)
                hidden_layer = self.enco(tt, self.hidden_vec * r)
                self.hidden_vec = z * hidden_layer + (1 - z) * self.hidden_vec

    def update(self, states, rewards, next_states, his_flag=True):
        if his_flag:
            self.calcuHistory()

        # 这个next_states还要经过一次计算

        # input = self.env.get_state_vec(ent, rel, target_e) # input a tensor
        r = self.resgate(states, self.hidden_vec)
        z = self.zgate(states, self.hidden_vec)
        hidden_layer = self.enco(states, self.hidden_vec * r)

        next_hidden = z * hidden_layer + (1 - z) * self.hidden_vec
        cat = torch.cat((states, next_hidden))
        y = self.policy_decoder(cat)
        y = F.relu(y)
        # y = F.softmax(y, dim=0)

        # y = self.acModel.update(y,actions,rewards,next_states,dones)
        policy, value = self.actor_critic(y)

        action_probs = policy.squeeze().detach().numpy()
        action = np.random.choice(np.arange(self.n_action), p=action_probs)

        # print(type(next_states),next_states)

        _, next_value = self.actor_critic(next_states)

        delta = rewards + 0.99 * next_value - value

        actor_loss = -torch.log(policy.squeeze()[action]) * delta
        critic_loss = torch.square(delta)

        total_loss = actor_loss + critic_loss

        self.optimizer.zero_grad()
        total_loss.backward(retain_graph=True)
        # total_loss.backward()
        self.optimizer.step()

        self.hidden_vec = next_hidden
        return action


if __name__ == '__main__':
    policy = Policy()
    done = False

    while not done:
        # ........
        list1 = [random.randint(1, 100000) for _ in range(10)]
        list2 = [random.randint(1, 100000) for _ in range(10)]
        state = torch.tensor(list1, dtype=torch.float)
        reward = 0
        next_state = torch.tensor(list2, dtype=torch.float)
        action = policy.update(state, reward, next_state)

        print("action= ", action)

        # exit()
