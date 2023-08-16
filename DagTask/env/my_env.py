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
        self.nodes_done = np.zeros(self.num_nodes, dtype=np.bool_)
        # 环境重置
        self.reset()
        self.get_action_space()

    def reset(self):  # 环境重置
        # 已完成节点数量
        self.nodes_done = np.zeros(self.num_nodes, dtype=np.bool_)
        # 是否完成
        self.is_done = False
        return self.get_state()

    '''
        获取节点state的tensor
        '''

    def get_state(self):
        return torch.tensor(self.nodes_done, dtype=torch.float32)

    '''
        获取动作空间
        '''

    def get_action_space(self):
        in_degrees = np.sum(self.adjacency_matrix, axis=0)
        out_degrees = np.sum(self.adjacency_matrix, axis=1)
        zero_in_degrees_nodes = np.where(in_degrees == 0)[0]
        zero_in_and_out_degrees_nodes = np.where((in_degrees == 0) & (out_degrees == 0))[0]
        action_space = list(set(zero_in_degrees_nodes) - set(zero_in_and_out_degrees_nodes))
        # 找到只有对角线为1的节点
        diagonal_nodes = np.where(np.diag(self.adjacency_matrix) == 1)[0]  # 对角线为1的节点
        only_diagonal_nodes = []
        for node in diagonal_nodes:
            if (sum(self.adjacency_matrix[node, :]) + sum(self.adjacency_matrix[:, node]) == 2):
                only_diagonal_nodes.append(node)
        action_space.extend(only_diagonal_nodes)
        action_space.sort()

        if not action_space:
            action_space = [self.num_nodes - 1]
        return action_space

    # def get_action_space(cur_adj_matrix, task_index):
    #     # task_index代表当前调度节点
    #     num_task = cur_adj_matrix.shape[0]
    #     # 生成调度候选集
    #     task_list = []
    #     for i in range(num_task):
    #         if sum(cur_adj_matrix[i, :]) != 0 or sum(cur_adj_matrix[:, i]) != 0:
    #             task_list.append(i)
    #     task_list.remove(task_index)
    #     print(task_list)  # [1, 2, 3]
    #     cur_adj_matrix[task_index, :] = 0
    #     action_space = []
    #     for j in task_list:
    #         if np.sum(cur_adj_matrix[:, j]) == 0:
    #             action_space.append(j)
    #     cur_adj_matrix[task_index, :] = 0
    #     return action_space

    def step(self, action):
        reward = 0
        done = False

        if not self.nodes_done[action]:
            self.nodes_done[action] = True
            # 将已调度节点的行置零
            self.adjacency_matrix[action, :] = 0
            cur_adj = self.adjacency_matrix
            # print(self.adjacency_matrix)
            if np.all(self.nodes_done):
                reward = 1
                done = True
        else:
            print('out of action_space')
        next_state = self.get_state()

        return next_state, reward, done


