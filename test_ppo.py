from Components.Modeler_Component_test import *
from Components.Adapter_Component import *
from Components.Policy import *
from collections import deque
from cfg import get_cfg
from GPO import Agent
import numpy as np

from scipy.stats import randint

fix_l = 0
fix_u = 17

def preprocessing(scenarios):
    scenario = scenarios
    if mode == 'txt':
        input_path = ["Data/Test/dataset{}/ship.txt".format(scenario),
                      "Data/Test/dataset{}/patrol_aircraft.txt".format(scenario),
                      "Data/Test/dataset{}/SAM.txt".format(scenario),
                      "Data/Test/dataset{}/SSM.txt".format(scenario),
                      "Data/Test/dataset{}/inception.txt".format(scenario)]
    else:
        input_path = "Data/Test/dataset{}/input_data.xlsx".format(scenario)

    data = Adapter(input_path=input_path,
                   mode=mode,
                   polar_chart=episode_polar_chart,
                   polar_chart_visualize=polar_chart_visualize)
    return data







def evaluation(agent, env):
    temp = random.uniform(fix_l, fix_u)
    agent_yellow = Policy(env, rule='rule2', temperatures=[temp, temp])
    done = False
    episode_reward = 0
    eval = False
    enemy_action_for_transition = [0] * len(env.enemies_fixed_list)
    friendly_action_for_transition = [0] * len(env.friendlies_fixed_list)
    while not done:
        if env.now % (decision_timestep) <= 0.00001:
            avail_action_blue, target_distance_blue, air_alert_blue = env.get_avail_actions_temp(side='blue')
            avail_action_yellow, target_distance_yellow, air_alert_yellow = env.get_avail_actions_temp(side='yellow')
            # edge_index_ssm_to_ship = env.get_ssm_to_ship_edge_index()
            # edge_index_ssm_to_ssm = env.get_ssm_to_ssm_edge_index()
            # edge_index_sam_to_ssm = env.get_sam_to_ssm_edge_index()
            # edge_index_ship_to_sam = env.get_ship_to_sam_edge_index()
            # edge_index_ship_to_enemy = env.get_ship_to_enemy_edge_index()
            # heterogeneous_edges = (edge_index_ssm_to_ship, edge_index_ssm_to_ssm, edge_index_sam_to_ssm, edge_index_ship_to_sam, edge_index_ship_to_enemy)
            # ship_feature = env.get_ship_feature()
            # missile_node_feature, node_cats = env.get_missile_node_feature()
            # action_feature = env.get_action_feature()
            actions_blue = list()
            for i in range(len(env.friendlies_fixed_list)):
                edge_index_ssm_to_ship = env.get_ssm_to_ship_edge_index(k = i)
                edge_index_ssm_to_ssm = env.get_ssm_to_ssm_edge_index(k = i)
                edge_index_sam_to_ssm = env.get_sam_to_ssm_edge_index(k = i)
                edge_index_ship_to_sam = env.get_ship_to_sam_edge_index(k = i)
                edge_index_ship_to_enemy = env.get_ship_to_enemy_edge_index(k = i)
                heterogeneous_edges = (edge_index_ssm_to_ship, edge_index_ssm_to_ssm, edge_index_sam_to_ssm, edge_index_ship_to_sam,edge_index_ship_to_enemy)
                ship_feature = env.get_ship_feature(k = i)
                missile_node_feature, node_cats = env.get_missile_node_feature(k = i)
                action_feature = env.get_action_feature(k = i)
                agent.eval_check(eval=True)
                action_blue, prob, mask, a_index = agent.sample_action(ship_feature, missile_node_feature,
                                                                       heterogeneous_edges, avail_action_blue[i],
                                                                       action_feature)
                actions_blue.append(action_blue)


            action_yellow = agent_yellow.get_action(avail_action_yellow, target_distance_yellow, air_alert_yellow)
            reward, win_tag, done, leakers = env.step(actions_blue, action_yellow)
            episode_reward += reward
        else:
            pass_transition = True
            actions_blue = list()
            for i in range(len(env.friendlies_fixed_list)):
                actions_blue.append([0, 0, 0, 0, 0, 0, 0, 0])
            env.step(action_blue=actions_blue, action_yellow=enemy_action_for_transition,pass_transition=pass_transition)

    return episode_reward, win_tag




if __name__ == "__main__":
    cfg = get_cfg()
    vessl_on = cfg.vessl
    if vessl_on == True:
        import vessl

        vessl.init()
        output_dir = "/output/"
        import os

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    else:
        print("시작")
        from torch.utils.tensorboard import SummaryWriter

        output_dir = "../output_susceptibility/"
        writer = SummaryWriter('./logs2')
        import os

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    import time

    """

    환경 시스템 관련 변수들

    """
    visualize = False  # 가시화 기능 사용 여부 / True : 가시화 적용, False : 가시화 미적용
    size = [600, 600]  # 화면 size / 600, 600 pixel
    tick = 500  # 가시화 기능 사용 시 빠르기
    n_step = cfg.n_step
    simtime_per_frame = cfg.simtime_per_frame
    decision_timestep = cfg.decision_timestep
    detection_by_height = False  # 고도에 의한
    mode = 'excel'  # 전처리 모듈 / 'excel' : input_data.xlsx 파일 적용, 'txt' "Data\ship.txt", "Data\patrol_aircraft.txt", "Data\SAM.txt", "Data\SSM.txt"를 적용
    rule = 'rule2'  # rule1 : 랜덤 정책 / rule2 : 거리를 기반 합리성에 기반한 정책(softmax policy)
    temperature = [10, 20]  # rule = 'rule2'인 경우만 적용 / 의사결정의 flexibility / 첫번째 index : 공중 위험이 낮은 상태, 두번째 index : 공중 위험이 높은 상태
    ciws_threshold = 1
    polar_chart_visualize = False
    scenarios = ['scenario1', 'scenario2', 'scenario3']
    lose_ratio = list()
    remains_ratio = list()
    polar_chart_scenario1 = [33, 29, 25, 33, 30, 30, 55, 27, 27, 35, 25, 30, 40]  # RCS의 polarchart 적용
    print(cfg)
    polar_chart = [polar_chart_scenario1]
    df_dict = {}
    # scenario = np.random.choice(scenarios)
    episode_polar_chart = polar_chart[0]
    records = list()
    datasets = [i for i in range(7, 31)]
    non_lose_ratio_list = []
    raw_data = list()
    for dataset in datasets:
        print("====dataset{}====".format(dataset))
        fitness_history = []
        data = preprocessing(dataset)
        t = 0
        env = modeler(data,
                      visualize=visualize,
                      size=size,
                      detection_by_height=detection_by_height,
                      tick=tick,
                      simtime_per_framerate=simtime_per_frame,
                      ciws_threshold=ciws_threshold,
                      action_history_step=cfg.action_history_step)

        agent = Agent(action_size=env.get_env_info()["n_actions"],
                      feature_size_ship=env.get_env_info()["ship_feature_shape"],
                      feature_size_missile=env.get_env_info()["missile_feature_shape"],
                      n_node_feature_missile=env.friendlies_fixed_list[0].air_tracking_limit +
                                             env.friendlies_fixed_list[0].air_engagement_limit +
                                             env.friendlies_fixed_list[0].num_m_sam +
                                             1,
                     node_embedding_layers_ship=list(eval(cfg.ship_layers)),
                     node_embedding_layers_missile=list(eval(cfg.missile_layers)),
                     n_representation_ship = cfg.n_representation_ship,
                     n_representation_missile = cfg.n_representation_missile,
                     n_representation_action = cfg.n_representation_action,

                     learning_rate=cfg.lr,
                     learning_rate_critic=cfg.lr_critic,
                     gamma=cfg.gamma,
                     lmbda=cfg.lmbda,
                     eps_clip = cfg.eps_clip,
                     K_epoch = cfg.K_epoch,
                     layers=list(eval(cfg.ppo_layers))
                     )
        load_file = "episode15450"
        agent.load_network(load_file+'.pt') # 2900, 1600
        reward_list = list()

        non_lose_ratio = 0
        non_lose_records = []
        seed = cfg.seed
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

        for e in range(cfg.n_test):
            env = modeler(data,
                          visualize=visualize,
                          size=size,
                          detection_by_height=detection_by_height,
                          tick=tick,
                          simtime_per_framerate=simtime_per_frame,
                          ciws_threshold=ciws_threshold,
                          action_history_step=cfg.action_history_step)
            episode_reward, win_tag = evaluation(agent, env)
            if win_tag != 'lose':
                non_lose_ratio += 1/cfg.n_test
                non_lose_records.append(1)
                raw_data.append([str(env.random_recording), 1])
            else:
                non_lose_records.append(0)
                raw_data.append([str(env.random_recording), 0])


            print(e, win_tag, np.mean(non_lose_records))

        non_lose_ratio_list.append(non_lose_ratio)
        df = pd.DataFrame(non_lose_ratio_list)
        df.to_csv("rev_ppo_result_{}_angle_{}.csv".format(load_file, cfg.inception_angle))
        df_raw = pd.DataFrame(raw_data)
        df_raw.to_csv("rev_raw_data_ppo_angle_{}.csv".format(cfg.inception_angle))

