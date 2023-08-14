'''
@Project ：AC 
@File    ：my_env.py
@IDE     ：PyCharm 
@Author  ：chaoyi yang
@Date    ：2023/8/11 16:23 
'''

import numpy as np
import torch
import torch.nn as nn


class Resources(object):

    def __init__(self, mec_process_capable,
                 mobile_process_capable, bandwidth_up=7.0, bandwidth_dl=7.0):
        self.mec_process_capble = mec_process_capable
        self.mobile_process_capable = mobile_process_capable
        self.mobile_process_avaliable_time = 0.0
        self.mec_process_avaliable_time = 0.0

        self.bandwidth_up = bandwidth_up
        self.bandwidth_dl = bandwidth_dl

    # 上传传输时间
    def up_transmission_cost(self, data):
        rate = self.bandwidth_up * (1024.0 * 1024.0 / 8.0)

        transmission_time = data / rate

        return transmission_time

    def reset(self):
        self.mec_process_avaliable_time = 0.0
        self.mobile_process_avaliable_time = 0.0

    # 接收时间
    def dl_transmission_cost(self, data):
        rate = self.bandwidth_dl * (1024.0 * 1024.0 / 8.0)
        transmission_time = data / rate

        return transmission_time

    # 本地处理时间
    def locally_execution_cost(self, data):
        return self._computation_cost(data, self.mobile_process_capable)

    # 边缘处理时间
    def mec_execution_cost(self, data):
        return self._computation_cost(data, self.mec_process_capble)

    # 计算公式
    def _computation_cost(self, data, processing_power):
        computation_time = data / processing_power

        return computation_time


'''
    定义环境
    '''


class env():
    def __init__(self, adjacency_matrix):  # 自定义参数
        # 传入邻接矩阵
        self.adjacency_matrix = adjacency_matrix
        # 获取节点数量
        self.num_nodes = len(adjacency_matrix)
        # 追踪节点被调度情况（是否全部调度完成）
        self.nodes_done = np.zeros(self.num_nodes, dtype=np.bool)
        # 环境重置
        self.reset()

    def reset(self):  # 环境重置
        # 当前节点
        self.current_node = 0
        # 已完成节点数量
        self.nodes_done = 0
        # 是否完成
        self.is_done = False
        return self.get_state_vec()

    '''
        获取节点state的tensor
        '''

    def get_state_vec(adj_matrix):
        state_vec = torch.tensor(adj_matrix, dtype=torch.float32).view(-1)
        return state_vec

    '''
        获取动作空间
        '''

    def get_action_space(cur_adj_matrix, task_index):
        # task_index代表当前调度节点
        num_task = cur_adj_matrix.shape[0]
        # 生成调度候选集
        task_list = []
        for i in range(num_task):
            if sum(cur_adj_matrix[i, :]) != 0 or sum(cur_adj_matrix[:, i]) != 0:
                task_list.append(i)
        task_list.remove(task_index)
        print(task_list)  # [1, 2, 3]
        cur_adj_matrix[task_index, :] = 0
        action_space = []
        for j in task_list:
            if np.sum(cur_adj_matrix[:, j]) == 0:
                action_space.append(j)
        cur_adj_matrix[task_index, :] = 0
        return action_space, cur_adj_matrix

    def is_done(adj_matrix, done):
        if sum(adj_matrix[:, :]) == 0:
            done = True
        else:
            done = False
        return done

    def step(self, action):
        reward = 0
        done = False

        if not self.nodes_done[action]:
            self.nodes_done[action] = True
            if np.all(self.nodes_done):
                reward = 1
                done = True
        else:
            print('out of action_space')
        next_state = self.get_state_vec()

        return next_state, reward, done


# 假设有5个节点
num_nodes = 5
# 创建布尔数组，表示节点的调度状态
nodes_done = np.zeros(num_nodes, dtype=np.bool_)
# 假设节点2和节点4已经被调度完成
nodes_done[2] = True
nodes_done[4] = True
if not nodes_done[1]:
    # 输出节点的调度状态
    print("Nodes done:", nodes_done)
    print("nodes_done[2]:", nodes_done[1])
else:
    print("nodes_done[2]:",nodes_done[1])