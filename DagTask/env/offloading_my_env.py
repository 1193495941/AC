"""
@Project ：AC 
@File    ：my_env.py
@IDE     ：PyCharm 
@Author  ：chaoyi yang
@Date    ：2023/8/11 16:23 
"""

import numpy as np
import torch
import pygraphviz as pgv


class Resources:
    """
    This class denotes the MEC server and Mobile devices (computation resources)

    Args:
        mec_process_capable: computation capacity of the MEC server
        mobile_process_capable: computation capacity of the mobile device
        bandwidth_up: wireless uplink bandwidth
        bandwidth_dl: wireless downlink bandwidth
    """

    def __init__(self, mec_process_capable, mobile_process_capable, bandwidth_up, bandwidth_dl):
        self.mec_process_capble = mec_process_capable
        self.mobile_process_capable = mobile_process_capable
        self.bandwidth_up = bandwidth_up
        self.bandwidth_dl = bandwidth_dl

    def up_transmission_cost(self, data):
        rate = self.bandwidth_up * (1024.0 * 1024.0 / 8.0)

        transmission_time = data / rate

        return transmission_time

    def dl_transmission_cost(self, data):
        rate = self.bandwidth_dl * (1024.0 * 1024.0 / 8.0)
        transmission_time = data / rate

        return transmission_time

    def locally_execution_cost(self, data):
        return self._computation_cost(data, self.mobile_process_capable)

    def mec_execution_cost(self, data):
        return self._computation_cost(data, self.mec_process_capble)

    def _computation_cost(self, data, processing_power):
        computation_time = data / processing_power

        return computation_time


# resource_cluster = Resources(mec_process_capable=(10 * 1024 * 1024),
#                              mobile_process_capable=(1.0 * 1024 * 1024),
#                              bandwidth_up=14.0,
#                              bandwidth_dl=14.0)


class TaskGraph:
    def __init__(self, graph_file_path):
        self.graph_file_path = graph_file_path
        self.graph = pgv.AGraph(graph_file_path)
        self.node_info = {}  # 存储节点信息的字典
        self.num_nodes = len(self.graph.nodes())  # 节点数量
        self.task_list = []  # 任务节点list

        for node in self.graph.nodes():
            node_id = int(node)
            self.node_info[node_id] = {
                'size': int(node.attr['size']),
                'expect_size': int(node.attr['expect_size']),
                'predecessors': [],
                'successors': []
            }

        for edge in self.graph.edges():
            source_id = int(edge[0])
            target_id = int(edge[1])
            self.node_info[source_id]['successors'].append(target_id)
            self.node_info[target_id]['predecessors'].append(source_id)

    #   获取node_info
    def get_node_info(self, node_id):
        return self.node_info.get(node_id)

    def get_node_size(self, node_id):
        node_size = self.get_node_info(node_id)['size']
        return node_size

    def get_node_expect_size(self, node_id):
        node_expect_size = self.get_node_info(node_id)['expect_size']
        return node_expect_size

    def get_node_successors(self, node_id):
        node_successors = self.get_node_info(node_id)['successors']
        return node_successors

    def get_node_predecessors(self, node_id):
        node_predecessors = self.get_node_info(node_id)['predecessors']
        return node_predecessors


class OffloadingEnvironment(Resources, TaskGraph):
    def __init__(self, mec_process_capable, mobile_process_capable, bandwidth_up, bandwidth_dl,
                 graph_file_paths):
        Resources.__init__(self, mec_process_capable, mobile_process_capable, bandwidth_up, bandwidth_dl)
        TaskGraph.__init__(self, graph_file_paths)
        self.cloud_available_time = 0.0
        self.ws_available_time = 0.0
        self.local_available_time = 0.0
        self.edge_available_time = 0.0
        self.current_time = 0.0
        self.length = self.num_nodes + 1
        # running time on local processor
        self.T_l = [0] * self.length
        # running time on sending channel
        self.T_ul = [0] * self.length
        # running time on receiving channel
        self.T_dl = [0] * self.length

        # finish time on cloud for each task_index
        self.FT_cloud = [0] * self.length
        # finish time on sending channel for each task_index
        self.FT_ws = [0] * self.length
        # finish time locally for each task_index
        self.FT_locally = [0] * self.length
        # finish time receiving channel for each task_index
        self.FT_wr = [0] * self.length

    #   将 offload_random20_{X} 文件夹内所有的.gz 添加到task_graph_list中
    def task_graph_list(self, graph_number, graph_file_paths):
        task_graph_list = []
        for i in range(graph_number):
            task_graph = TaskGraph(graph_file_paths + str(i) + '.gz')
            task_graph_list.append(task_graph)
        return task_graph_list

    def reset(self):
        self.cloud_available_time = 0.0
        self.ws_available_time = 0.0
        self.local_available_time = 0.0  # 目前本地端时间
        self.edge_available_time = 0.0  # 目前边缘端时间
        self.current_time = 0.0  # 当前卸载的最大时间,是本地端和边缘端的最大时间

        # running time on local processor
        self.T_l = [0] * self.length
        # running time on sending channel
        self.T_ul = [0] * self.length
        # running time on receiving channel
        self.T_dl = [0] * self.length
        # finish time on cloud for each task_index
        self.FT_cloud = [0] * self.length
        # finish time on sending channel for each task_index
        self.FT_ws = [0] * self.length
        # finish time locally for each task_index
        self.FT_locally = [0] * self.length
        # finish time receiving channel for each task_index
        self.FT_wr = [0] * self.length

    #   生成状态向量
    def offloading_state_list(self):
        state_vector_list = ['occupy']
        for i in range(self.num_nodes):
            task_id = i + 1
            task = self.get_node_info(task_id)
            local_cost = task['size'] / self.mobile_process_capable
            up_cost = self.up_transmission_cost(task['size'])
            mec_cost = task['size'] / self.mec_process_capble
            down_cost = self.dl_transmission_cost(task['expect_size'])
            #   节点嵌入向量
            task_embedding_vector = [task_id, local_cost, up_cost,
                                     mec_cost, down_cost]
            #   后继节点
            succs_task_index_set = []
            task_successors = task['successors']
            for succs in task_successors:
                succs_task_index_set.append(succs)
            while len(succs_task_index_set) < 6:
                succs_task_index_set.append(-1.0)
            #   节点前继
            pre_task_index_set = []
            task_predecessors = task['predecessors']
            for pre in task_predecessors:
                pre_task_index_set.append(pre)
            while len(pre_task_index_set) < 6:
                pre_task_index_set.append(-1.0)

            succs_task_index_set = succs_task_index_set[0:6]
            pre_task_index_set = pre_task_index_set[0:6]
            #   本地边端系统状态
            system_set = []
            while len(system_set) < 2:
                system_set.append(0.0)
            #   历史动作信息
            history_set = []
            while len(history_set) < 20:
                history_set.append(-1.0)

            state_vector = task_embedding_vector + pre_task_index_set + succs_task_index_set + \
                           system_set + history_set
            state_vector_list.append(state_vector)

        return state_vector_list

    def offloading_step(self, offloading_action, task):
        # locally scheduling
        if offloading_action == 0:
            if len(self.get_node_predecessors(task)) != 0:
                start_time = max(self.local_available_time,
                                 max([max(self.FT_locally[j], self.FT_wr[j]) for j in
                                      self.get_node_predecessors(task)]))
            else:
                start_time = self.local_available_time

            self.T_l[task] = self.locally_execution_cost(self.get_node_size(task))
            self.FT_locally[task] = start_time + self.T_l[task]

            self.local_available_time = self.FT_locally[task]
            task_finish_time = self.FT_locally[task]
            reward = self.current_time - max(task_finish_time, self.edge_available_time)
            self.current_time = max(task_finish_time, self.edge_available_time)

            return reward, self.current_time

        # mec scheduling
        else:
            if len(self.get_node_predecessors(task)) != 0:
                ws_start_time = max(self.ws_available_time,
                                    max([max(self.FT_locally[j], self.FT_ws[j]) for j in
                                         self.get_node_predecessors(task)]))

                self.T_ul[task] = self.up_transmission_cost(self.get_node_size(task))
                ws_finish_time = ws_start_time + self.T_ul[task]
                self.FT_ws[task] = ws_finish_time
                self.ws_available_time = ws_finish_time  # 传输后的时间

                cloud_start_time = max(self.cloud_available_time,
                                       max([max(self.FT_ws[task], self.FT_cloud[j]) for j in
                                            self.get_node_predecessors(task)]))
                cloud_finish_time = cloud_start_time + self.mec_execution_cost(self.get_node_size(task))
                self.FT_cloud[task] = cloud_finish_time
                self.cloud_available_time = cloud_finish_time

                wr_start_time = self.FT_cloud[task]
                self.T_dl[task] = self.dl_transmission_cost(self.get_node_expect_size(task))
                wr_finish_time = wr_start_time + self.T_dl[task]
                self.FT_wr[task] = wr_finish_time

            else:
                ws_start_time = self.ws_available_time
                self.T_ul[task] = self.up_transmission_cost(self.get_node_size(task))
                ws_finish_time = ws_start_time + self.T_ul[task]
                self.FT_ws[task] = ws_finish_time  # 任务传输给边端后的时间
                self.ws_available_time = ws_finish_time

                cloud_start_time = max(self.cloud_available_time, self.FT_ws[task])
                cloud_finish_time = cloud_start_time + self.mec_execution_cost(self.get_node_size(task))
                self.FT_cloud[task] = cloud_finish_time
                self.cloud_available_time = cloud_finish_time  # 任务在边端完成的时间

                wr_start_time = self.FT_cloud[task]
                self.T_dl[task] = self.dl_transmission_cost(self.get_node_expect_size(task))
                wr_finish_time = wr_start_time + self.T_dl[task]
                self.FT_wr[task] = wr_finish_time  # 任务在边端传回本地后完成的时间

            task_finish_time = wr_finish_time
            self.edge_available_time = task_finish_time
            reward = self.current_time - max(self.local_available_time, task_finish_time)
            self.current_time = max(self.local_available_time, task_finish_time)

            return reward, self.current_time


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

    def step(self, action):

        if not self.nodes_done[action]:
            self.nodes_done[action] = True
        else:
            print('out of action_space')
        next_state = self.get_state()

        return self.nodes_done


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
        bad_time = self.locally_execution_cost(TaskGraph(file_path).sum_data_size())

        # running time on local processor   # 本地运行时间
        T_l = [0] * TaskGraph(file_path).num_nodes
        # running time on sending channel   # 传输到边缘时间
        T_ul = [0] * TaskGraph(file_path).num_nodes
        # running time on receiving channel # 本地接收时间
        T_dl = [0] * TaskGraph(file_path).num_nodes

        # finish time on cloud for each task_index    # 边端完成时间
        FT_cloud = [0] * TaskGraph(file_path).num_nodes
        # finish time on sending channel for each task_index  #
        FT_ws = [0] * TaskGraph(file_path).num_nodes
        # finish time locally for each task_index     # 本地完成时间
        FT_locally = [0] * TaskGraph(file_path).num_nodes
        # finish time receiving channel for each task_index   #
        FT_wr = [0] * TaskGraph(file_path).num_nodes  # 接收

        # 卸载前的最大时间
        behind_time = max(self.FT_local, self.FT_edge)
        if behind_time == self.FT_local:
            behind_time = behind_time
            pre_time = behind_time + remain_time1  # 卸载前的总时间
        else:
            if len(TaskGraph(file_path).get_pre_task(task_id) != 0):
                pre_local_time = max(self.local_available_time,
                                     max([max(FT_locally[j], FT_wr[j]) for j in
                                          TaskGraph(file_path).get_pre_task(task_id)])) + remain_time1
                pre_time = max(behind_time, pre_local_time)
            else:
                pre_local_time = self.FT_local + remain_time1
                pre_time = max(behind_time, pre_local_time)

        # 本地端
        if unload_action == 0:
            if len(TaskGraph(file_path).get_pre_task(task_id) != 0):
                start_time = max(self.local_available_time,
                                 max([max(FT_locally[j], FT_wr[j]) for j in
                                      TaskGraph(file_path).get_pre_task(task_id)]))
            else:
                start_time = self.local_available_time
            T_l_copy = T_l.copy()
            T_l_copy[task_id] = self.locally_execution_cost(TaskGraph(file_path).get_data_size(task_id))
            FT_locally_copy = FT_locally.copy()
            FT_locally_copy[task_id] = start_time + T_l_copy[task_id]

            copy_local_available_time = FT_locally_copy[task_id]
            copy_task_finish_time = FT_locally_copy[task_id]
            copy_FT_local = FT_locally_copy[task_id]
            self.dag_time = [copy_FT_local, self.FT_edge]
            self.max_local_edge = max(copy_FT_local, self.FT_edge)

            if self.max_local_edge == copy_FT_local:
                after_time = self.max_local_edge + remain_time2
            else:
                if len(TaskGraph(file_path).get_pre_task(task_id) != 0):
                    after_local_time = max(copy_local_available_time,
                                           max([max(FT_locally_copy[j], FT_wr[j]) for j in
                                                TaskGraph(file_path).get_pre_task(task_id)])) + remain_time2
                    after_time = max(self.max_local_edge, after_local_time)

                else:
                    after_local_time = copy_FT_local + remain_time2
                    after_time = max(self.max_local_edge, after_local_time)

            reward2 = pre_time - after_time

            if reward2 == 0:
                if len(TaskGraph(file_path).get_pre_task(task_id) != 0):
                    # 传输 开始时间
                    ws_start_time = max(self.ws_available_time,
                                        max([max(FT_locally[j], FT_ws[j]) for j in
                                             TaskGraph(file_path).get_pre_task(task_id)]))
                    # 当前节点传输到边缘端的时间
                    T_ul_copy = T_ul.copy()
                    T_ul_copy[task_id] = self.up_transmission_cost(TaskGraph(file_path).get_data_size(task_id))
                    ws_finish_time = ws_start_time + T_ul_copy[task_id]
                    FT_ws_copy = FT_ws.copy()
                    FT_ws_copy[task_id] = ws_finish_time

                    copy_ws_available_time = ws_finish_time

                    cloud_start_time = max(self.cloud_available_time,
                                           max([max(FT_ws_copy[task_id], FT_cloud[j]) for j in
                                                TaskGraph(file_path).get_pre_task(task_id)]))
                    cloud_finish_time = cloud_start_time + self.mec_execution_cost(
                        TaskGraph(file_path).get_data_size(task_id))
                    FT_cloud_copy = FT_cloud.copy()
                    FT_cloud_copy[task_id] = cloud_finish_time

                    copy_cloud_available_time = cloud_finish_time

                    wr_start_time = FT_cloud_copy[task_id]
                    T_dl_copy = T_dl.copy()
                    T_dl_copy[task_id] = self.dl_transmission_cost(TaskGraph(file_path).get_data_size(task_id))
                    wr_finish_time = wr_start_time + T_dl_copy[task_id]
                    FT_wr_copy = FT_wr.copy()
                    FT_wr_copy[task_id] = wr_finish_time

                    copy_FT_edge = wr_finish_time
                    copy_max_local_edge = max(self.FT_local, copy_FT_edge)

                else:
                    ws_start_time = self.ws_available_time
                    T_ul_copy = T_ul.copy()
                    T_ul_copy[task_id] = self.up_transmission_cost(TaskGraph(file_path).get_data_size(task_id))
                    ws_finish_time = ws_start_time + T_ul_copy[task_id]
                    FT_ws_copy = FT_ws.copy()
                    FT_ws_copy[task_id] = ws_finish_time

                    cloud_start_time = max(self.cloud_available_time, FT_ws_copy[task_id])
                    cloud_finish_time = cloud_start_time + self.mec_execution_cost(
                        TaskGraph(file_path).get_data_size(task_id))
                    FT_cloud_copy = FT_cloud.copy()
                    FT_cloud_copy[task_id] = cloud_finish_time
                    copy_cloud_available_time = cloud_finish_time

                    wr_start_time = FT_cloud_copy[task_id]
                    T_dl_copy = T_dl.copy()
                    T_dl_copy[task_id] = self.dl_transmission_cost(TaskGraph(file_path).get_data_size(task_id))
                    wr_finish_time = wr_start_time + T_dl_copy[task_id]
                    FT_wr_copy = FT_wr.copy()
                    FT_wr_copy[task_id] = wr_finish_time

                    copy_FT_edge = wr_finish_time
                    copy_max_local_edge = max(self.FT_local, copy_FT_edge)

                copy_task_finish_time = wr_finish_time
                if copy_max_local_edge == self.FT_local:
                    v_after_time = copy_max_local_edge + remain_time2
                else:
                    if len(TaskGraph(file_path).get_pre_task(task_id) != 0):
                        after_local_time = max(self.local_available_time,
                                               max([max(FT_locally[j], FT_wr[j]) for j in
                                                    TaskGraph(file_path).get_pre_task(task_id)])) + remain_time2
                        v_after_time = max(copy_max_local_edge, after_local_time)
                    else:
                        after_local_time = self.FT_local + remain_time2
                        v_after_time = max(copy_max_local_edge, after_local_time)

                reward2 = v_after_time - after_time

            # 再更新
            T_l[task_id] = self.locally_execution_cost(TaskGraph(file_path).get_data_size(task_id))
            FT_locally[task_id] = start_time + T_l[task_id]

            self.local_available_time = FT_locally[task_id]
            self.task_finish_time = FT_locally[task_id]
            self.FT_local = FT_locally[task_id]
            self.dag_time = [self.FT_local, self.FT_edge]
            self.max_local_edge = max(self.FT_local, self.FT_edge)

            next_state = self.get_state()

            return reward2, next_state, after_time, self.dag_time

        # 边缘端
        else:
            if len(TaskGraph(file_path).get_pre_task(task_id) != 0):
                # 传输 开始时间
                ws_start_time = max(self.ws_available_time,
                                    max([max(FT_locally[j], FT_ws[j]) for j in
                                         TaskGraph(file_path).get_pre_task(task_id)]))
                # 当前节点传输到边缘端的时间
                T_ul[task_id] = self.up_transmission_cost(TaskGraph(file_path).get_data_size(task_id))
                ws_finish_time = ws_start_time + T_ul[task_id]
                FT_ws[task_id] = ws_finish_time

                self.ws_available_time = ws_finish_time

                cloud_start_time = max(self.cloud_available_time,
                                       max([max(FT_ws[task_id], FT_cloud[j]) for j in
                                            TaskGraph(file_path).get_pre_task(task_id)]))
                cloud_finish_time = cloud_start_time + self.mec_execution_cost(
                    TaskGraph(file_path).get_data_size(task_id))
                FT_cloud[task_id] = cloud_finish_time

                self.cloud_available_time = cloud_finish_time

                wr_start_time = FT_cloud[task_id]
                T_dl[task_id] = self.dl_transmission_cost(TaskGraph(file_path).get_data_size(task_id))
                wr_finish_time = wr_start_time + T_dl[task_id]
                FT_wr[task_id] = wr_finish_time

                self.FT_edge = wr_finish_time
                self.dag_time = [self.FT_local, self.FT_edge]
                self.max_local_edge = max(self.FT_local, self.FT_edge)

            else:
                ws_start_time = self.ws_available_time
                T_ul[task_id] = self.up_transmission_cost(TaskGraph(file_path).get_data_size(task_id))
                ws_finish_time = ws_start_time + T_ul[task_id]
                FT_ws[task_id] = ws_finish_time

                cloud_start_time = max(self.cloud_available_time, FT_ws[task_id])
                cloud_finish_time = cloud_start_time + self.mec_execution_cost(
                    TaskGraph(file_path).get_data_size(task_id))
                FT_cloud[task_id] = cloud_finish_time
                self.cloud_available_time = cloud_finish_time

                wr_start_time = FT_cloud[task_id]
                T_dl[task_id] = self.dl_transmission_cost(TaskGraph(file_path).get_data_size(task_id))
                wr_finish_time = wr_start_time + T_dl[task_id]
                FT_wr[task_id] = wr_finish_time

                self.FT_edge = wr_finish_time
                self.dag_time = [self.FT_local, self.FT_edge]
                self.max_local_edge = max(self.FT_local, self.FT_edge)

            self.task_finish_time = wr_finish_time

            if self.max_local_edge == self.FT_local:
                after_time = self.max_local_edge + remain_time2
            else:
                if len(TaskGraph(file_path).get_pre_task(task_id) != 0):
                    after_local_time = max(self.local_available_time,
                                           max([max(FT_locally[j], FT_wr[j]) for j in
                                                TaskGraph(file_path).get_pre_task(task_id)])) + remain_time2
                    after_time = max(self.max_local_edge, after_local_time)
                else:
                    after_local_time = self.FT_local + remain_time2
                    after_time = max(self.max_local_edge, after_local_time)
            reward2 = pre_time - after_time

            self.dag_time = [self.FT_local, self.FT_edge]
            next_state = self.get_state()

            return reward2, next_state, after_time, self.dag_time
