"""
    main
"""
from env.my_env import schedule_env
from env.my_env import unload_env
from model import *

if __name__ == "__main__":
    # 初始化参数
    num_epochs = 100
    learning_rate = 0.1

    for i in range(3):
        # 读取 .gv 文件并解析为图形对象
        file_path = "./env/data/meta_offloading_20/offload_random20_1/random.20.{}.gv".format(i)
        print(file_path)

        # 生成初始化并预处理的邻接矩阵
        init_adjacency_matrix = taskGraph(file_path).gen_adjacency_matrix()

        # 假设DAG是由节点和依赖关系构成的，依赖关系表示为一个邻接矩阵
        adjacency_matrix = np.copy(init_adjacency_matrix)

        s_env = schedule_env(adjacency_matrix)  # 调度环境
        u_env = unload_env(edge_process_capable=(2.5 * 1024 * 1024),
                           local_process_capable=(1.0 * 1024 * 1024), bandwidth_up=5.0, bandwidth_dl=5.0)  # 卸载环境
        action_space = s_env.get_action_space()
        model = ActorCritic(state1_dim=s_env.num_nodes, state2_dim=2, hidden_dim=128, action_dim=s_env.num_nodes,
                            unload_dim=2)

        # 训练Actor-Critic模型
        epochs, epoch_rewards = train_actor_critic(s_env, u_env, model, num_epochs, learning_rate, file_path,
                                                   init_adjacency_matrix)

        model_path = './trained_model_pth/model{}.pth'.format(i)
        torch.save(model.state_dict(), model_path)

    '''
            epochs和reward 图
            '''

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
