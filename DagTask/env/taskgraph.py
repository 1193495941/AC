'''
@Project ：AC 
@File    ：taskgraph.py
@IDE     ：PyCharm 
@Author  ：chaoyi yang
@Date    ：2023/8/17 16:13 
'''
import numpy as np
import re



class taskGraph(object):
    def __init__(self, file_path):
        self.file_path = file_path
        self.gen_adjacency_matrix()
        self.get_data_size()

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
        num_nodes = max([int(node) for node in nodes])

        # 初始化邻接矩阵
        gen_adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=int)

        # 填充邻接矩阵
        for src, dst in edges:
            src_idx = int(src) - 1
            dst_idx = int(dst) - 1
            gen_adjacency_matrix[src_idx, dst_idx] = 1

        # 对邻接矩阵进行预处理
        out_degrees = np.sum(gen_adjacency_matrix, axis=1)
        zero_out_degrees_nodes = np.where(out_degrees == 0)[0]
        gen_adjacency_matrix[zero_out_degrees_nodes, zero_out_degrees_nodes] = 1
        init_adjacency_matrix = gen_adjacency_matrix

        return init_adjacency_matrix

    # 任务节点的 前继节点
    def get_pre_task(self, adjacency_matrix, task_id):
        self.adj = adjacency_matrix
        self.task_id = task_id
        pre_task_sets = np.where(self.adj[:, self.task_id] == 1)[0]
        return pre_task_sets

    # 获取节点的size大小
    def get_data_size(self):
        with open(self.file_path, "r") as f:
            gv_content = f.read()
        # 提取节点信息
        node_pattern = r"(\d+) \[size=\"(\d+)\",.*\]"
        nodes = re.findall(node_pattern, gv_content)

        # 将提取的节点信息转换为字典，其中键为节点索引，值为节点的size（作为时延）
        node_delay_map = {int(node[0]): int(node[1]) for node in nodes}

        return node_delay_map

    def get_bad_time(self):
        self.node_dalay_map = self.get_data_size()
        bad_time = sum(self.node_dalay_map.values())
        return bad_time
