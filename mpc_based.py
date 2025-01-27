import numpy as np
import torch, random
from cfg import get_cfg


from Components.Modeler_Component_test import *
from Components.Adapter_Component import *
from Components.Policy import *
from collections import deque
from GDN import Agent
from scipy.optimize import minimize
from scipy.optimize import Bounds

fix_l = 0
fix_u = 17

 # 현재 까지 5가 최고임

# 5에서 0.9
# 15에서 0.74

def action_changer(a, avail_action):
    terminated = True
    avail_action = avail_action[0]
    while terminated:
        if avail_action[a] == True:
            terminated = False
        else:
            a -= 1
    return a


# MPC Optimization objective function
def objective_function(u, env, package = True):
    n_eval = 5
    episode_reward = 0
    for _ in range(n_eval):

        env = deepcopy(env)
        u = np.array(u, dtype = np.int)
        u = u.tolist()
        temp = random.uniform(fix_l, fix_u)
        agent_yellow = Policy(env, rule='rule2', temperatures=[temp, temp])
        enemy_action_for_transition = [0] * len(env.enemies_fixed_list)
        for t in range(prediction_horizon):
            if env.now % (decision_timestep) <= 0.00001:
                avail_action_blue, target_distance_blue, air_alert_blue = env.get_avail_actions_temp(side='blue')
                avail_action_yellow, target_distance_yellow, air_alert_yellow = env.get_avail_actions_temp(side='yellow')
                #action_blue = agent_blue.get_action(avail_action_blue, target_distance_blue, air_alert_blue)
                action_yellow = agent_yellow.get_action(avail_action_yellow, target_distance_yellow, air_alert_yellow)
                action_blue = u[t]
                action_blue = action_changer(action_blue, avail_action_blue)
                reward, win_tag, done, leakers = env.step([action_blue], action_yellow, rl=False)
                episode_reward += reward
                if (done == True):
                    if ((win_tag == 'draw') or (win_tag == 'win')):
                        episode_reward += 1000
                    else:
                        episode_reward -= 1000
                    break

            else:
                pass_transition = True
                actions_blue = list()
                for i in range(len(env.friendlies_fixed_list)):
                    actions_blue.append([0, 0, 0, 0, 0, 0, 0, 0])
                env.step(action_blue=actions_blue, action_yellow=enemy_action_for_transition,pass_transition=pass_transition)
    return -episode_reward  # Minimize the negative total reward (maximize reward)
def constraint_function(x, avail_action_blue, timestep):
    # 해제약 함수 정의
    # avail_action_blue와 비교하여 timestep에 해당하는 해제약을 만족하는지 확인하고 만족하지 않으면 큰 값을 반환
    #print(x, timestep)
    #print(avail-)
    #avail_action_blue[timestep]
    #for t in range(prediction_horizon):
    #print(avail_action_blue[timestep])
    if avail_action_blue[timestep][0][int(x[timestep])]== False:
        return 1000

    # for i, avail in enumerate(avail_action_blue[timestep]):
    #     print(avail)
    #     if avail and not x[i]:
    #         return -1.0  # 큰 값 반환
    return 0.0
def mathematical_optimization(env, avail_action_blue):
    # 초기 추정치 설정
    action_size = np.arange(len(avail_action_blue[0])).tolist()
    p = (np.array(avail_action_blue, dtype=np.float_) / np.array(avail_action_blue, dtype=np.float_).sum()).reshape(-1)
    initial_guess = [np.random.choice(action_size, p=p) for _ in range(prediction_horizon)]

    constraints = []
    avail_action_blue = [avail_action_blue if i/prediction_horizon <= 0.3 else [[True]*len(avail_action_blue[0])] for i in range(prediction_horizon)]
    for t in range(prediction_horizon):
        constraints.append({'type': 'eq', 'fun': constraint_function, 'args': (avail_action_blue, t)})
    result = minimize(objective_function,initial_guess, method='SLSQP',
                      args = (env,),
                      constraints=constraints)

    optimal_control_sequence = result.x

    return optimal_control_sequence


def simulated_annealing_mpc(prediction_horizon, population_size, num_iterations, initial_temperature, temperature_decay, mutation_rate, env, avail_action_blue):
    action_size = np.arange(len(avail_action_blue[0])).tolist()
    p = (np.array(avail_action_blue, dtype=np.float_) / np.array(avail_action_blue, dtype=np.float_).sum()).reshape(-1)

    def generate_initial_solution():
        return [np.random.choice(action_size, p=p) for _ in range(prediction_horizon)]

    current_solution = generate_initial_solution()
    best_solution = deepcopy(current_solution)
    current_temperature = initial_temperature

    for iteration in range(num_iterations):
        env_temp = deepcopy(env)

        current_fitness = objective_function(current_solution, env_temp, package=False)
        best_fitness = objective_function(best_solution, env_temp, package=False)

        new_solution = deepcopy(current_solution)
        if np.random.rand() < mutation_rate:
            mutation_index = np.random.randint(prediction_horizon)
            mutation_value = np.random.choice(action_size, p=p)
            new_solution[mutation_index] = mutation_value

        new_fitness = objective_function(new_solution, env_temp, package=False)

        if new_fitness < current_fitness or np.random.rand() < np.exp((current_fitness - new_fitness) / current_temperature):
            current_solution = new_solution
            current_fitness = new_fitness

            if new_fitness < best_fitness:
                best_solution = new_solution

        current_temperature *= temperature_decay

    return best_solution


# Genetic Algorithm for MPC optimization
def genetic_algorithm_mpc(population_size, num_generations, mutation_rate, env, avail_action_blue):
    #rint(avail_action_blue)
    action_size = np.arange(len(avail_action_blue[0])).tolist()
    p = (np.array(avail_action_blue, dtype = np.float_)/np.array(avail_action_blue, dtype = np.float_).sum()).reshape(-1)
    #print(p.shape, len(action_size))
    population = [ [np.random.choice(action_size, p = p) for _ in range(prediction_horizon)] for _ in range(population_size)]
    for generation in range(num_generations):

        env_temp = deepcopy(env)

        fitness = [objective_function(u, env_temp, package = False) for u in population]
        sorted_indices = np.argsort(fitness)
        sorted_population = [population[i] for i in sorted_indices]

        # Elitism: Keep the best solution from the previous generation
        best_solution = sorted_population[0]

        # Roulette wheel selection (select parents based on fitness)
        num_parents = population_size - 1
        parents = sorted_population[:num_parents]

        # Crossover: Create new solutions by combining parents
        crossover_point = np.random.randint(1, prediction_horizon)
        offspring = [np.concatenate((parents[i][:crossover_point],
                                     parents[i+1][crossover_point:])) for i in range(num_parents-1)]
        offspring.append(np.concatenate((parents[num_parents - 1][crossover_point:], parents[0][:crossover_point])))

        # Mutation: Introduce random changes in the offspring
        for i in range(num_parents):
            if np.random.rand() < mutation_rate:
                mutation_index = np.random.randint(prediction_horizon)
                mutation_value = np.random.choice(action_size, p = p)
                offspring[i][mutation_index] = mutation_value

        population = [best_solution] + offspring
    #print(best_solution)
    return best_solution


def particle_swarm_optimization(num_particles, num_iterations, env, avail_action_blue):
    action_size = np.arange(len(avail_action_blue[0])).tolist()
    p = (np.array(avail_action_blue, dtype=np.float_) / np.array(avail_action_blue, dtype=np.float_).sum()).reshape(-1)
    particles = np.random.choice(action_size, size=(num_particles, prediction_horizon), p=p)
    velocities = np.zeros_like(particles)

    global_best_solution = None
    global_best_fitness = float('inf')

    for iteration in range(num_iterations):
        for i in range(num_particles):
            env_temp = deepcopy(env)
            fitness = objective_function(particles[i],
                                         env_temp,
                                         package = False)
            if fitness < global_best_fitness:
                global_best_fitness = fitness
                global_best_solution = particles[i]

            local_best_fitness = fitness
            local_best_solution = particles[i]

            # Update particle velocity and position
            inertia_weight = 0.5
            cognitive_weight = 1.5
            social_weight = 1.5
            r1 = np.random.random(particles[i].shape)
            r2 = np.random.random(particles[i].shape)

            velocities[i] = (inertia_weight * velocities[i] +
                             cognitive_weight * r1 * (local_best_solution - particles[i]) +
                             social_weight * r2 * (global_best_solution - particles[i]))

            particles[i] += velocities[i]

            # Apply bounds to particle position
            particles[i] = np.clip(particles[i], 0, len(action_size) - 1)

    return global_best_solution


def preprocessing(scenario):
    if mode == 'txt':
        input_path = ["Data/Test/dataset{}/ship.txt".format(scenario),
                      "Data/Test/dataset{}/patrol_aircraft.txt".format(scenario),
                      "Data/Test/dataset{}/SAM.txt".format(scenario),
                      "Data/Test/dataset{}/SSM.txt".format(scenario),
                      "Data/Test/dataset{}/inception.txt".format(scenario)]
    else:
        input_path = "Data\input_data.xlsx"

    data = Adapter(input_path=input_path,
                   mode=mode,
                   polar_chart=episode_polar_chart,
                   polar_chart_visualize=polar_chart_visualize)
    return data



def evaluation(env):
    temp = random.uniform(fix_l, fix_u)
    agent_yellow = Policy(env, rule='rule2', temperatures=[temp, temp])
    done = False
    episode_reward = 0

    enemy_action_for_transition = [0] * len(env.enemies_fixed_list)
    overtime = None

    while not done:
        if env.now % (decision_timestep) <= 0.00001:
            avail_action_blue, target_distance_blue, air_alert_blue = env.get_avail_actions_temp(side='blue')
            avail_action_yellow, target_distance_yellow, air_alert_yellow = env.get_avail_actions_temp(side='yellow')
            action_yellow = agent_yellow.get_action(avail_action_yellow, target_distance_yellow, air_alert_yellow)


            # optimal_control_sequence = genetic_algorithm_mpc(population_size=20, num_generations=30, mutation_rate=0.3, env=env, avail_action_blue=avail_action_blue)
            # optimal_control_sequence = np.array(optimal_control_sequence, dtype=np.int)
            optimal_control_sequence = mathematical_optimization(env, avail_action_blue)
            optimal_control_sequence = np.array(optimal_control_sequence, dtype=np.int)
            action_blue = optimal_control_sequence[0]

            reward, win_tag, done, leakers = env.step([action_blue], action_yellow, rl=False)
            episode_reward += reward

        else:
            pass_transition = True
            actions_blue = list()
            for i in range(len(env.friendlies_fixed_list)):
                actions_blue.append([0, 0, 0, 0, 0, 0, 0, 0])
            env.step(action_blue=actions_blue, action_yellow=enemy_action_for_transition,pass_transition=pass_transition)

    return episode_reward, win_tag, leakers, overtime


if __name__ == "__main__":

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
    visualize =False # 가시화 기능 사용 여부 / True : 가시화 적용, False : 가시화 미적용
    size = [600, 600]  # 화면 size / 600, 600 pixel
    tick = 500  # 가시화 기능 사용 시 빠르기
    n_step = cfg.n_step
    simtime_per_frame = cfg.simtime_per_frame
    decision_timestep = cfg.decision_timestep
    detection_by_height = False  # 고도에 의한
    num_iteration = cfg.num_episode  # 시뮬레이션 반복횟수
    mode = 'txt'  # 전처리 모듈 / 'excel' : input_data.xlsx 파일 적용, 'txt' "Data\ship.txt", "Data\patrol_aircraft.txt", "Data\SAM.txt", "Data\SSM.txt"를 적용
    rule = 'rule2'  # rule1 : 랜덤 정책 / rule2 : 거리를 기반 합리성에 기반한 정책(softmax policy)

    ciws_threshold = 1
    polar_chart_visualize = False
    #3scenarios = ['scenario1', 'scenario2', 'scenario3']
    lose_ratio = list()
    remains_ratio = list()
    polar_chart_scenario1 = [33, 29, 25, 33, 30, 30, 55, 27, 27, 35, 25, 30, 40]  # RCS의 polarchart 적용
    polar_chart = [polar_chart_scenario1]
    df_dict = {}
    episode_polar_chart = polar_chart[0]
    records = list()
    data = preprocessing(1)
    t = 0


    eval_lose_ratio = list()
    eval_win_ratio = list()
    lose_ratio = list()
    win_ratio = list()
    reward_list = list()

    eval_lose_ratio1 = list()
    eval_win_ratio1 = list()
    print("noise", cfg.with_noise)
    non_lose_rate = list()

    n = 500
    non_lose_rate = list()

    prediction_horizon = 5

    for j in range(n):
        env = modeler(data,
                      visualize=visualize,
                      size=size,
                      detection_by_height=detection_by_height,
                      tick=tick,
                      simtime_per_framerate=simtime_per_frame,
                      ciws_threshold=ciws_threshold,
                      action_history_step=cfg.action_history_step,
                      interval_constant_blue=[cfg.interval_constant_blue, cfg.interval_constant_blue]
                      )
        episode_reward, win_tag, leakers, overtime = evaluation(env)
        if win_tag == 'draw' or win_tag == 'win':
            non_lose_rate.append(1)
        print('전', win_tag, episode_reward, env.now, overtime, np.sum(non_lose_rate)/(j+1))
