"""
@Project ：AC 
@File    ：my_env.py
@IDE     ：PyCharm 
@Author  ：chaoyi yang
@Date    ：2023/8/11 16:23 
"""

import numpy as np
import torch
import re
import math


class taskGraph(object):
    def __init__(self, file_path):
        self.file_path = file_path
        self.gen_adjacency_matrix()
        self.get_all_data_size()
        self.init_adjacency_matrix
        self.num_nodes

    # 生成邻接矩阵并做预处理
    def gen_adjacency_matrix(self):
        with open(self.file_path, "r") as f:
            gv_content = f.read()

        # 提取节点和边的信息
        node_pattern = r"(\d+) \[.*\]"
        edge_pattern = r"(\d+) -> (\d+) .*"
        nodes = re.findall(node_pattern, gv_content)
        edges = re.findall(edge_pattern, gv_content)

        # 确定节点的数量
        self.num_nodes = max([int(node) for node in nodes])

        # 初始化邻接矩阵
        gen_adjacency_matrix = np.zeros((self.num_nodes, self.num_nodes), dtype=int)

        # 填充邻接矩阵
        for src, dst in edges:
            src_idx = int(src) - 1
            dst_idx = int(dst) - 1
            gen_adjacency_matrix[src_idx, dst_idx] = 1

        # 对邻接矩阵进行预处理
        out_degrees = np.sum(gen_adjacency_matrix, axis=1)
        zero_out_degrees_nodes = np.where(out_degrees == 0)[0]
        gen_adjacency_matrix[zero_out_degrees_nodes, zero_out_degrees_nodes] = 1
        self.init_adjacency_matrix = gen_adjacency_matrix

        return self.init_adjacency_matrix

    # 任务节点的 前继节点
    def get_pre_task(self, task_id):
        self.adj = self.init_adjacency_matrix
        self.task_id = task_id
        pre_task_sets = np.where(self.adj[:, self.task_id] == 1)[0]
        return pre_task_sets

    # 获取节点的size大小
    def get_data_size(self, task_id):
        with open(self.file_path, "r") as f:
            gv_content = f.read()
        # 提取节点信息
        node_pattern = r"(\d+) \[size=\"(\d+)\",.*\]"
        nodes = re.findall(node_pattern, gv_content)

        # 将提取的节点信息转换为字典，其中键为节点索引，值为节点的size（作为时延）
        node_delay_map = {int(node[0]): int(node[1]) for node in nodes}
        data_size = node_delay_map.get(task_id + 1)

        return data_size

    def get_all_data_size(self):
        with open(self.file_path, "r") as f:
            gv_content = f.read()
        # 提取节点信息
        node_pattern = r"(\d+) \[size=\"(\d+)\",.*\]"
        nodes = re.findall(node_pattern, gv_content)

        # 将提取的节点信息转换为字典，其中键为节点索引，值为节点的size（作为时延）
        node_delay_map = {int(node[0]): int(node[1]) for node in nodes}
        return node_delay_map

    def sum_data_size(self):
        self.node_delay_map = self.get_all_data_size()
        data_size = sum(self.node_delay_map.values())

        return data_size


'''
    定义环境
    '''


class schedule_env:
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
            if sum(self.adjacency_matrix[node, :]) + sum(self.adjacency_matrix[:, node]) == 2:
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
        reward1 = 0
        done = False

        if not self.nodes_done[action]:
            self.nodes_done[action] = True
            # 将已调度节点的行置零
            self.adjacency_matrix[action, :] = 0
            # cur_adj = self.adjacency_matrix
            # print(self.adjacency_matrix)
            if np.all(self.nodes_done):
                reward1 = 0
                done = True
        else:
            print('out of action_space')
        next_state = self.get_state()

        return next_state, reward1, done


class unload_env(object):

    def __init__(self, edge_process_capable,
                 local_process_capable, bandwidth_up=10.0, bandwidth_dl=10.0):
        self.edge_process_capable = edge_process_capable
        self.local_process_capable = local_process_capable
        self.bandwidth_up = bandwidth_up  # 7.0
        self.bandwidth_dl = bandwidth_dl  # 7.0
        self.FT_local = 0
        self.FT_edge = 0

        self.local_available_time = 0.0
        self.cloud_available_time = 0.0
        self.ws_available_time = 0.0
        self.task_finish_time = 0.0
        self.min_local_edge = 0.0
        self.max_local_edge = 0.0
        self.current_FT = 0.0

        self.dag_time = [self.FT_local, self.FT_edge]
        self.reset()

    # 上传传输时间
    def up_transmission_cost(self, data):
        rate = self.bandwidth_up * (1024.0 * 1024.0 / 8.0)
        transmission_time = data / rate
        return transmission_time

    # 接收时间
    def dl_transmission_cost(self, data):
        rate = self.bandwidth_dl * (1024.0 * 1024.0 / 8.0)
        transmission_time = data / rate
        return transmission_time

    # 本地处理时间
    def locally_execution_cost(self, data):
        return self._computation_cost(data, self.local_process_capable)

    # 边缘处理时间
    def mec_execution_cost(self, data):
        return self._computation_cost(data, self.edge_process_capable)

    # 计算公式
    def _computation_cost(self, data, processing_power):
        computation_time = data / processing_power
        return computation_time

    # 获取状态  state2 = [local_time,FT_edge]
    def get_state(self):
        return torch.tensor(self.dag_time, dtype=torch.float32)

    def get_action_space(self):
        action_space = [0, 1]
        return action_space

    def reset(self):
        self.FT_local = 0.0
        self.FT_edge = 0.0
        self.local_available_time = 0.0
        self.cloud_available_time = 0.0
        self.ws_available_time = 0.0
        self.task_finish_time = 0.0
        self.max_local_edge = 0.0
        self.min_local_edge = 0.0
        self.current_FT = 0.0
        self.dag_time = [0, 0]
        state = self.get_state()
        return state

    def step(self, file_path, unload_action, task_id, remain_time1, remain_time2):
        # 最坏时间
        bad_time = self.locally_execution_cost(taskGraph(file_path).sum_data_size())

        assert abs(bad_time - 1212) < 1

        # running time on local processor   # 本地运行时间
        T_l = [0] * taskGraph(file_path).num_nodes
        # running time on sending channel   # 传输到边缘时间
        T_ul = [0] * taskGraph(file_path).num_nodes
        # running time on receiving channel # 本地接收时间
        T_dl = [0] * taskGraph(file_path).num_nodes

        # finish time on cloud for each task_index    # 边端完成时间
        FT_cloud = [0] * taskGraph(file_path).num_nodes
        # finish time on sending channel for each task_index  #
        FT_ws = [0] * taskGraph(file_path).num_nodes
        # finish time locally for each task_index     # 本地完成时间
        FT_locally = [0] * taskGraph(file_path).num_nodes
        # finish time receiving channel for each task_index   #
        FT_wr = [0] * taskGraph(file_path).num_nodes

        # 卸载前的最大时间
        behind_time = max(self.FT_local, self.FT_edge)
        if behind_time == self.FT_local:
            behind_time = behind_time
            pre_time = behind_time + remain_time1  # 卸载前的总时间
        else:
            if len(taskGraph(file_path).get_pre_task(task_id) != 0):
                pre_local_time = max(self.local_available_time,
                                     max([max(FT_locally[j], FT_wr[j]) for j in
                                          taskGraph(file_path).get_pre_task(task_id)])) + remain_time1
                pre_time = max(behind_time, pre_local_time)
            else:
                pre_local_time = self.FT_local + remain_time1
                pre_time = max(behind_time, pre_local_time)

        # 本地端
        if unload_action == 0:
            if len(taskGraph(file_path).get_pre_task(task_id) != 0):
                start_time = max(self.local_available_time,
                                 max([max(FT_locally[j], FT_wr[j]) for j in
                                      taskGraph(file_path).get_pre_task(task_id)]))
            else:
                start_time = self.local_available_time

            T_l[task_id] = self.locally_execution_cost(taskGraph(file_path).get_data_size(task_id))
            FT_locally[task_id] = start_time + T_l[task_id]
            self.local_available_time = FT_locally[task_id]

            self.task_finish_time = FT_locally[task_id]

            self.FT_local = FT_locally[task_id]
            self.max_local_edge = max(self.FT_local, self.FT_edge)

            if self.max_local_edge == self.FT_local:
                after_time = self.max_local_edge + remain_time2
            else:
                if len(taskGraph(file_path).get_pre_task(task_id) != 0):
                    after_local_time = max(self.local_available_time,
                                           max([max(FT_locally[j], FT_wr[j]) for j in
                                                taskGraph(file_path).get_pre_task(task_id)])) + remain_time2
                    after_time = max(self.max_local_edge, after_local_time)
                else:
                    after_local_time = self.FT_local + remain_time1
                    after_time = max(self.max_local_edge, after_local_time)
            # print(f'{task_id},本地pre time:{pre_time}')
            # print(f'{task_id},本地after time:{after_time}')
            reward2 = pre_time - after_time
            # print(f'本地reward2:{reward2}')

            # reward2 = (1 - (self.max_local_edge - self.min_local_edge) / self.max_local_edge)
            self.dag_time = [self.FT_local, self.FT_edge]
            # print('dag_time:',self.dag_time)
            next_state2 = self.get_state()
            # print('local_reward:', reward2)
            return reward2, next_state2, after_time, self.max_local_edge

        # 边缘端
        else:
            if len(taskGraph(file_path).get_pre_task(task_id) != 0):
                # 传输 开始时间
                ws_start_time = max(self.ws_available_time,
                                    max([max(FT_locally[j], FT_ws[j]) for j in
                                         taskGraph(file_path).get_pre_task(task_id)]))
                # 当前节点传输到边缘端的时间
                T_ul[task_id] = self.up_transmission_cost(taskGraph(file_path).get_data_size(task_id))
                ws_finish_time = ws_start_time + T_ul[task_id]
                FT_ws[task_id] = ws_finish_time

                self.ws_available_time = ws_finish_time

                cloud_start_time = max(self.cloud_available_time,
                                       max([max(FT_ws[task_id], FT_cloud[j]) for j in
                                            taskGraph(file_path).get_pre_task(task_id)]))
                cloud_finish_time = cloud_start_time + self.mec_execution_cost(
                    taskGraph(file_path).get_data_size(task_id))
                FT_cloud[task_id] = cloud_finish_time

                self.cloud_available_time = cloud_finish_time

                wr_start_time = FT_cloud[task_id]
                T_dl[task_id] = self.dl_transmission_cost(taskGraph(file_path).get_data_size(task_id))
                wr_finish_time = wr_start_time + T_dl[task_id]
                FT_wr[task_id] = wr_finish_time

                self.FT_edge = wr_finish_time
                self.max_local_edge = max(self.FT_local, self.FT_edge)

            else:
                ws_start_time = self.ws_available_time
                T_ul[task_id] = self.up_transmission_cost(taskGraph(file_path).get_data_size(task_id))
                ws_finish_time = ws_start_time + T_ul[task_id]
                FT_ws[task_id] = ws_finish_time

                cloud_start_time = max(self.cloud_available_time, FT_ws[task_id])
                cloud_finish_time = cloud_start_time + self.mec_execution_cost(
                    taskGraph(file_path).get_data_size(task_id))
                FT_cloud[task_id] = cloud_finish_time
                self.cloud_available_time = cloud_finish_time

                wr_start_time = FT_cloud[task_id]
                T_dl[task_id] = self.dl_transmission_cost(taskGraph(file_path).get_data_size(task_id))
                wr_finish_time = wr_start_time + T_dl[task_id]
                FT_wr[task_id] = wr_finish_time

                self.FT_edge = wr_finish_time
                self.max_local_edge = max(self.FT_local, self.FT_edge)

            self.task_finish_time = wr_finish_time

        if self.max_local_edge == self.FT_local:
            after_time = self.max_local_edge + remain_time2
        else:
            if len(taskGraph(file_path).get_pre_task(task_id) != 0):
                after_local_time = max(self.local_available_time,
                                       max([max(FT_locally[j], FT_wr[j]) for j in
                                            taskGraph(file_path).get_pre_task(task_id)])) + remain_time2
                after_time = max(self.max_local_edge, after_local_time)
            else:
                after_local_time = self.FT_local + remain_time2
                after_time = max(self.max_local_edge, after_local_time)
        # print(f'{task_id},边端pre time:{pre_time}')
        # print(f'{task_id},边端after time:{after_time}')

        reward2 = pre_time - after_time
        # print(f'边端reward2:{reward2}')

        self.current_FT = max(self.task_finish_time, self.current_FT)
        # self.current_FT = max(self.FT_local, self.FT_edge)

        self.min_local_edge = min(self.FT_local, self.FT_edge)

        # reward2 = (1 - (self.max_local_edge - self.min_local_edge) / self.max_local_edge)
        self.dag_time = [self.FT_local, self.FT_edge]
        # print('dag_time:', self.dag_time)
        next_state2 = self.get_state()
        # print('edge_reward:',reward2)
        return reward2, next_state2, after_time, bad_time
