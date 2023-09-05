"""
    main
"""
import os.path
from env.offloading_my_env import unload_env
from model import *

if __name__ == "__main__":
    # 初始化参数
    federal_num_epochs = 10
    num_epochs = 1000
    learning_rate = 0.001

    for f_epochs in range(federal_num_epochs):
        # 每个分布式模型进行训练
        '''
        【模型的预训练】
            '''
        for i in range(3):
            # 读取 .gv 文件并解析为图形对象
            file_path = "./env/data/meta_offloading_20/offload_random20_1/random.20.{}.gv".format(i)
            print(file_path)

            # 生成初始化并预处理的邻接矩阵
            init_adjacency_matrix = taskGraph(file_path).gen_adjacency_matrix()

            # 假设DAG是由节点和依赖关系构成的，依赖关系表示为一个邻接矩阵
            adjacency_matrix = np.copy(init_adjacency_matrix)

            s_env = schedule_env(adjacency_matrix)  # 调度环境
            u_env = unload_env(edge_process_capable=(10 * 1024 * 1024),
                               local_process_capable=(1.0 * 1024 * 1024), bandwidth_up=14.0, bandwidth_dl=14.0)  # 卸载环境
            action_space = s_env.get_action_space()
            model = ActorCritic(state1_dim=s_env.num_nodes, state2_dim=2, hidden_dim=128, action_dim=s_env.num_nodes,
                                unload_dim=2)

            model_path = './trained_model_pth/model{}.pth'.format(i)
            updated_model_path = './updated_model_pth/model{}.pth'.format(i)

            if not os.path.exists(model_path):
                # 如果路径不存在,则自训练Actor-Critic模型,生成模型参数
                epochs, epoch_rewards = train_actor_critic(s_env, u_env, model, num_epochs,
                                                           learning_rate, file_path,
                                                           init_adjacency_matrix)
                print(f'federal epochs:{f_epochs}-------正在预训练model{i}-------')
                torch.save(model.state_dict(), model_path)
                print(f'federal epochs:{f_epochs}-------model{i}预训练完成-------')
            else:
                # 如果路径存在,则使用updated_model_pth相对应的model{}的模型参数进行更新
                if not os.path.exists(updated_model_path):
                    # 如果updated_model_pth里没有聚合模型则break
                    print(f'federal epochs:{f_epochs},model{i}未聚合')
                    break
                else:
                    # 如果文件存在,则更新并训练;然后再保存模型参数到trained_model_path
                    model.load_state_dict(torch.load(updated_model_path))
                    epochs, epoch_rewards = train_actor_critic(s_env, u_env, model, num_epochs,
                                                               learning_rate, file_path,
                                                               init_adjacency_matrix)
                    print(f'federal epochs:{f_epochs},model{i}正在使用云端模型参数训练')
                    torch.save(model.state_dict(), model_path)
        '''
        【模型的聚合】
            '''
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
            # print(loaded_models[i].state_dict())

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
                averaged_param = sum(weighted_params)
                aggregated_state_dict[name] = averaged_param

        # 更新模型参数
        for model in loaded_models:
            model.load_state_dict(aggregated_state_dict)
        print(f'federal epochs:{f_epochs},在云端完成了模型聚合')

        # 保存更新后的模型参数
        for i, model in enumerate(loaded_models):
            torch.save(model.state_dict(), './updated_model_pth/model{}.pth'.format(i))
            print(f'federal epochs:{f_epochs},model{i}已存放到updated文件夹')





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
