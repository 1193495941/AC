'''
@Project ：AC 
@File    ：avg.py
@IDE     ：PyCharm 
@Author  ：chaoyi yang
@Date    ：2023/8/22 09:37 
'''
import torch
from model import *
from env.my_env import schedule_env

# 加载已训练好的模型和图结构
loaded_state_dicts = []
loaded_models = []
file_paths = []
nodes = []

for i in range(3):
    file_path = "./env/data/meta_offloading_20/offload_random20_1/random.20.{}.gv".format(i)
    file_paths.append(file_path)
    # 生成初始化并预处理的邻接矩阵
    init_adjacency_matrix = taskGraph(file_path).gen_adjacency_matrix()
    # 假设DAG是由节点和依赖关系构成的，依赖关系表示为一个邻接矩阵
    adjacency_matrix = np.copy(init_adjacency_matrix)
    s_env = schedule_env(adjacency_matrix)  # 调度环境
    num_nodes = s_env.num_nodes  # 节点数量
    nodes.append(num_nodes)
    # 定义模型
    model = ActorCritic(state1_dim=num_nodes, state2_dim=2, hidden_dim=128, action_dim=num_nodes, unload_dim=2)
    model.load_state_dict(torch.load('./trained_model_pth/model{}.pth'.format(i)))
    loaded_models.append(model)
    print(loaded_models[i].state_dict())

total_nodes = sum(nodes)  # 总节点数量
weights = [nodes / total_nodes for nodes in nodes]  # 权重
new_model = ActorCritic(state1_dim=total_nodes, state2_dim=2,
                        hidden_dim=128, action_dim=total_nodes,
                        unload_dim=2)  # 新模型

aggregated_state_dict = {}
for k in range(len(loaded_models)):
    for name, _ in loaded_models[k].state_dict().items():
        # weighted_parms 是个list
        weighted_params = [model.state_dict()[name] * weight for model, weight in zip(loaded_models, weights)]
        # print(f'name{name}')
        averaged_param = sum(weighted_params)
        aggregated_state_dict[name] = averaged_param

# 更新模型参数
for model in loaded_models:
    model.load_state_dict(aggregated_state_dict)

# 保存更新后的模型参数
for i, model in enumerate(loaded_models):
    torch.save(model.state_dict(), './updated_model_pth/model{}.pth'.format(i))

# for name, _ in loaded_models[0].state_dict():
#     weighted_params = [model.state_dict()[name] * weight for model, weight in zip(loaded_models, weights)]
#     print(f'name:{name}')
#     averaged_param = sum(weighted_params)
#     aggregated_state_dict[name] = averaged_param

# for param_new, params in zip(new_model.parameters(), zip(*[model.parameters() for model in loaded_models])):
#     print(f'new model parameters:{param_new}')
#     print(f'parms:{params}')
# param_new.data.copy_(sum(p * w for p, w in zip(params, weights)))
