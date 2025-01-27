from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import CosineAnnealingLR
from ada_hessian import AdaHessian
import torch.cuda.amp as amp
import torch
import torch.nn as nn
import torch.nn.functional as F
import gc
import torch.optim as optim
import random
from collections import deque
from torch.distributions import Categorical
import numpy as np

from GAT.model import GAT
from GAT.layers import device
from copy import deepcopy
from GTN.utils import _norm
from GTN.model_fastgtn import FastGTNs
from scipy.sparse import csr_matrix
from collections import OrderedDict
from NoisyLinear import NoisyLinear

from cfg import get_cfg

cfg = get_cfg()


# import torch.cuda.OutOfMemoryError

def weight_init_xavier_uniform(submodule):
    if isinstance(submodule, torch.nn.Conv2d):
        torch.nn.init.xavier_uniform_(submodule.weight)
        submodule.bias.data.fill_(0.01)
    if isinstance(submodule, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(submodule.weight)
    elif isinstance(submodule, torch.nn.BatchNorm2d):
        submodule.weight.data.fill_(1.0)
        submodule.bias.data.zero_()


class DQNNetwork(nn.Module):
    def __init__(self, state_action_size, layers=[8,12]):
        super(DQNNetwork, self).__init__()
        self.state_action_size = state_action_size
        self.NN_sequential = OrderedDict()
        self.fc = nn.Linear(state_action_size, layers[0])
        self.fcn = OrderedDict()
        last_layer = layers[0]
        for i in range(1, len(layers)):
            layer = layers[i]
            if i <= len(layers)-2:
                self.fcn['linear{}'.format(i)] = nn.Linear(last_layer, layer)
                self.fcn['activation{}'.format(i)] = nn.ELU()
                last_layer = layer
            #else:
        self.forward_cal = nn.Sequential(self.fcn)
        self.output = nn.Linear(last_layer, 1)


    def forward(self, x):

        x = self.fc(x)
        x = F.elu(x)
        x = self.forward_cal(x)
        q = self.output(x)
        return q






class DuelingDQN(nn.Module):
    def __init__(self):
        super(DuelingDQN, self).__init__()

    def forward(self, V, A, mask, past_action=None, training=False):
        # v의 shape : batch_size x 1
        # a의 shape : batch_size x action size
        if (past_action == None) and (training == False):
            A = A.masked_fill(mask == 0, float('-inf'))
            zeros = torch.zeros_like(A)
            ones = torch.ones_like(A)
            nA = torch.where(A == float('-inf'), zeros, ones).sum()
            mean_A = torch.where(A == float('-inf'), zeros, A).sum()
            mean_A = mean_A / nA
            Q = V + A - mean_A

        if (past_action != None) and (training == True):
            # print(A.shape, mask.shape)

            mask = mask.squeeze(1)
            A = A.masked_fill(mask == 0, float('-inf'))
            zeros = torch.zeros_like(A)
            ones = torch.ones_like(A)
            nA = torch.where(A == float('-inf'), zeros, ones).sum(dim=1)
            mean_A = torch.where(A == float('-inf'), zeros, A).sum(dim=1)
            mean_A = mean_A / nA
            Q = V + past_action - mean_A.unsqueeze(1)
        if (past_action == None) and (training == True):
            mask = mask.squeeze(1)
            A = A.masked_fill(mask == 0, float('-inf'))

            zeros = torch.zeros_like(A)
            ones = torch.ones_like(A)

            nA = torch.where(A == float('-inf'), zeros, ones).sum(dim=1)
            mean_A = torch.where(A == float('-inf'), zeros, A).sum(dim=1)

            mean_A = mean_A / nA
            Q = V + A - mean_A.unsqueeze(1)

        return Q


class VDN(nn.Module):

    def __init__(self):
        super(VDN, self).__init__()

    def forward(self, q_local):
        return torch.sum(q_local, dim=1)


class Network(nn.Module):
    def __init__(self, obs_and_action_size, hidden_size_q, action_size):
        super(Network, self).__init__()
        self.obs_and_action_size = obs_and_action_size
        self.fcn_1 = nn.Linear(obs_and_action_size, hidden_size_q + 10)
        self.fcn_2 = nn.Linear(hidden_size_q + 10, hidden_size_q - 5)

        self.fcn_3 = nn.Linear(hidden_size_q - 5, hidden_size_q - 20)

        self.fcn_4 = nn.Linear(hidden_size_q - 20, hidden_size_q - 40)

        self.fcn_5 = nn.Linear(hidden_size_q - 40, action_size)
        # self.fcn_5 = nn.Linear(int(hidden_size_q/8), action_size)
        torch.nn.init.xavier_uniform_(self.fcn_1.weight)
        torch.nn.init.xavier_uniform_(self.fcn_2.weight)
        torch.nn.init.xavier_uniform_(self.fcn_3.weight)
        torch.nn.init.xavier_uniform_(self.fcn_4.weight)
        torch.nn.init.xavier_uniform_(self.fcn_5.weight)
        # torch.nn.init.xavier_uniform_(self.fcn_5.weight)

    def forward(self, obs_and_action):
        if obs_and_action.dim() == 1:
            obs_and_action = obs_and_action.unsqueeze(0)
        # print(obs_and_action.dim())

        x = F.elu(self.fcn_1(obs_and_action))
        x = F.elu(self.fcn_2(x))
        x = F.elu(self.fcn_3(x))
        x = F.elu(self.fcn_4(x))
        q = self.fcn_5(x)
        # q = self.fcn_5(x)
        return q


class NodeEmbedding(nn.Module):
    def __init__(self, feature_size, n_representation_obs, layers=[20, 30, 40]):
        super(NodeEmbedding, self).__init__()
        self.feature_size = feature_size
        self.linears = OrderedDict()
        last_layer = self.feature_size
        for i in range(len(layers)):
            layer = layers[i]
            if i <= len(layers) - 2:
                self.linears['linear{}'.format(i)] = nn.Linear(last_layer, layer)
                self.linears['activation{}'.format(i)] = nn.ELU()
                last_layer = layer
            else:
                self.linears['linear{}'.format(i)] = nn.Linear(last_layer, n_representation_obs)

        self.node_embedding = nn.Sequential(self.linears)
        # print(self.node_embedding)
        self.node_embedding.apply(weight_init_xavier_uniform)

    def forward(self, node_feature, missile=False):
        # print(node_feature.shape)
        node_representation = self.node_embedding(node_feature)
        return node_representation


class Replay_Buffer:
    def __init__(self, buffer_size, batch_size, n_node_feature_missile, n_node_feature_enemy, action_size, n_step,
                 per_alpha):
        self.buffer = deque()
        self.alpha = per_alpha
        self.step_count_list = list()
        for _ in range(18):
            self.buffer.append(deque(maxlen=buffer_size))
        self.buffer_size = buffer_size
        self.n_node_feature_missile = n_node_feature_missile
        self.n_node_feature_enemy = n_node_feature_enemy
        # self.agent_id = np.eye(self.num_agent).tolist()
        self.one_hot_actions = np.eye(action_size).tolist()
        self.batch_size = batch_size
        self.step_count = 0
        self.rewards_store = list()
        self.n_step = n_step

    def pop(self):
        self.buffer.pop()

    def memory(self,
               node_feature_missile,
               ship_feature,
               edge_index_missile,
               action,
               reward,

               done,
               avail_action,

               status,
               action_feature,
               action_features,
               heterogeneous_edges,
               action_index,
               node_cats
               ):

        self.buffer[1].append(node_feature_missile)
        self.buffer[2].append(ship_feature)

        self.buffer[3].append(edge_index_missile)
        self.buffer[4].append(action)

        self.buffer[5].append(list(reward))
        self.buffer[6].append(list(done))

        self.buffer[7].append(avail_action)
        self.buffer[8].append(status)
        self.buffer[9].append(np.sum(status))

        if len(self.buffer[10]) == 0:
            self.buffer[10].append(0.5)
        else:
            self.buffer[10].append(max(self.buffer[10]))

        self.buffer[13].append(action_feature)
        self.buffer[14].append(action_features)
        self.buffer[15].append(heterogeneous_edges)
        self.buffer[16].append(action_index)
        self.buffer[17].append(node_cats)

        if self.step_count < self.buffer_size:
            self.step_count_list.append(self.step_count)
            self.step_count += 1

        # print(len(self.buffer[7]), len(self.buffer[10]), len(self.step_count_list))

    # ship_features
    def generating_mini_batch(self, datas, batch_idx, cat):

        for s in batch_idx:
            if cat == 'node_feature_missile':
                yield datas[1][s]
            if cat == 'ship_features':
                yield datas[2][s]
            if cat == 'edge_index_missile':
                yield torch.sparse_coo_tensor(datas[3][s],
                                              torch.ones(torch.tensor(datas[3][s]).shape[1]),
                                              (self.n_node_feature_missile, self.n_node_feature_missile)).to_dense()
            if cat == 'action':
                yield datas[4][s]
            if cat == 'reward':
                yield datas[5][s]
            if cat == 'done':
                yield datas[6][s]

            if cat == 'node_feature_missile_next':
                yield datas[1][s + self.n_step]
            if cat == 'ship_features_next':
                yield datas[2][s + self.n_step]

            if cat == 'edge_index_missile_next':
                yield torch.sparse_coo_tensor(datas[3][s + self.n_step],
                                              torch.ones(torch.tensor(datas[3][s + self.n_step]).shape[1]),
                                              (self.n_node_feature_missile, self.n_node_feature_missile)).to_dense()
            if cat == 'avail_action':
                yield datas[7][s]
            if cat == 'avail_action_next':
                yield datas[7][s + self.n_step]
            if cat == 'status':
                yield datas[8][s]
            if cat == 'status_next':
                yield datas[8][s + self.n_step]

            if cat == 'priority':
                yield datas[10][s]
            if cat == 'action_feature':
                yield datas[13][s]

            if cat == 'action_features':
                # test
                yield datas[14][s]

            if cat == 'action_features_next':
                yield datas[14][s + self.n_step]

            if cat == 'heterogeneous_edges':
                yield datas[15][s]
            if cat == 'heterogeneous_edges_next':
                yield datas[15][s + self.n_step]
            if cat == 'action_index':
                yield datas[16][s]

            if cat == 'node_cats':
                yield datas[17][s]
            if cat == 'node_cats':
                yield datas[17][s + self.n_step]

    def update_transition_priority(self, batch_index, delta):

        copied_delta_store = deepcopy(list(self.buffer[10]))
        delta = np.abs(delta).reshape(-1) + np.min(copied_delta_store)
        priority = np.array(copied_delta_store).astype(float)

        batch_index = batch_index.astype(int)
        priority[batch_index] = delta
        self.buffer[10] = deque(priority, maxlen=self.buffer_size)

    def sample(self):
        step_count_list = self.step_count_list[:]
        sampled_batch_idx = np.random.choice(step_count_list[:-self.n_step], size=self.batch_size)
        node_feature_missile = self.generating_mini_batch(self.buffer, sampled_batch_idx, cat='node_feature_missile')
        node_features_missile = list(node_feature_missile)

        edge_index_missile = None  # self.generating_mini_batch(self.buffer, sampled_batch_idx, cat='edge_index_missile')
        edge_indices_missile = None  # list(edge_index_missile)

        node_feature_missile_next = self.generating_mini_batch(self.buffer, sampled_batch_idx,
                                                               cat='node_feature_missile_next')
        node_features_missile_next = list(node_feature_missile_next)

        edge_index_missile_next = None  # self.generating_mini_batch(self.buffer, sampled_batch_idx, cat='edge_index_missile_next')
        edge_indices_missile_next = None

        action = self.generating_mini_batch(self.buffer, sampled_batch_idx, cat='action')
        actions = list(action)

        ship_features = self.generating_mini_batch(self.buffer, sampled_batch_idx, cat='ship_features')
        ship_features = list(ship_features)

        ship_features_next = self.generating_mini_batch(self.buffer, sampled_batch_idx, cat='ship_features_next')
        ship_features_next = list(ship_features_next)

        reward = self.generating_mini_batch(self.buffer, sampled_batch_idx, cat='reward')
        rewards = list(reward)

        done = self.generating_mini_batch(self.buffer, sampled_batch_idx, cat='done')
        dones = list(done)

        avail_action = self.generating_mini_batch(self.buffer, sampled_batch_idx, cat='avail_action')
        avail_actions = list(avail_action)

        avail_action_next = self.generating_mini_batch(self.buffer, sampled_batch_idx, cat='avail_action_next')
        avail_actions_next = list(avail_action_next)

        status = self.generating_mini_batch(self.buffer, sampled_batch_idx, cat='status')
        status = list(status)

        status_next = self.generating_mini_batch(self.buffer, sampled_batch_idx, cat='status_next')
        status_next = list(status_next)

        priority = self.generating_mini_batch(self.buffer, sampled_batch_idx, cat='priority')
        priority = list(priority)

        action_feature = self.generating_mini_batch(self.buffer, sampled_batch_idx, cat='action_feature')
        action_feature = list(action_feature)

        action_features = self.generating_mini_batch(self.buffer, sampled_batch_idx, cat='action_features')
        action_features = list(action_features)

        action_features_next = self.generating_mini_batch(self.buffer, sampled_batch_idx, cat='action_features_next')
        action_features_next = list(action_features_next)

        heterogenous_edges = self.generating_mini_batch(self.buffer, sampled_batch_idx, cat='heterogeneous_edges')
        heterogenous_edges = list(heterogenous_edges)

        heterogenous_edges_next = self.generating_mini_batch(self.buffer, sampled_batch_idx,
                                                             cat='heterogeneous_edges_next')
        heterogenous_edges_next = list(heterogenous_edges_next)

        action_index = self.generating_mini_batch(self.buffer, sampled_batch_idx,
                                                  cat='action_index')
        action_index = list(action_index)

        node_cats = self.generating_mini_batch(self.buffer, sampled_batch_idx,
                                               cat='node_cats')
        node_cats = list(node_cats)

        node_cats_next = self.generating_mini_batch(self.buffer, sampled_batch_idx,
                                                    cat='node_cats_next')
        node_cats_next = list(node_cats_next)

        return node_features_missile, \
               ship_features, \
               edge_indices_missile, \
               actions, \
               rewards, \
               dones, \
               node_features_missile_next, \
               ship_features_next, \
               edge_indices_missile_next, \
               avail_actions, \
               avail_actions_next, \
               status, \
               status_next, \
               priority, \
               sampled_batch_idx,\
               action_feature, \
               action_features, \
               action_features_next, heterogenous_edges, heterogenous_edges_next, action_index, node_cats, node_cats_next


class Agent:
    def __init__(self,
                 num_agent,
                 feature_size_ship,
                 feature_size_missile,
                 feature_size_enemy,
                 feature_size_action,

                 iqn_layers,

                 node_embedding_layers_ship,
                 node_embedding_layers_missile,
                 node_embedding_layers_enemy,
                 node_embedding_layers_action,

                 n_multi_head,
                 n_representation_ship,
                 n_representation_missile,
                 n_representation_enemy,
                 n_representation_action,

                 hidden_size_enemy,
                 hidden_size_comm,

                 dropout,
                 action_size,
                 buffer_size,
                 batch_size,
                 learning_rate,
                 gamma,
                 GNN,
                 teleport_probability,
                 gtn_beta,
                 n_node_feature_missile,
                 n_node_feature_enemy,
                 n_step,
                 beta,
                 per_alpha,
                 iqn_layer_size,
                 iqn_N,
                 n_cos,
                 num_nodes,
                 layers

                 ):
        from cfg import get_cfg
        cfg = get_cfg()
        self.n_step = n_step
        self.num_agent = num_agent
        self.num_nodes = num_nodes
        self.feature_size_ship = feature_size_ship
        self.feature_size_missile = feature_size_missile

        self.n_multi_head = n_multi_head
        self.teleport_probability = teleport_probability
        self.beta = beta

        self.n_representation_ship = n_representation_ship
        self.n_representation_missile = n_representation_missile
        self.n_representation_enemy = n_representation_enemy

        self.action_size = action_size
        self.dummy_action = torch.tensor([[i for i in range(action_size)] for _ in range(batch_size)]).to(device)
        self.dropout = dropout
        self.gamma = gamma
        self.agent_id = np.eye(self.num_agent).tolist()
        self.agent_index = [i for i in range(self.num_agent)]
        self.max_norm = 10

        self.buffer_size = buffer_size
        self.batch_size = batch_size

        self.buffer = Replay_Buffer(self.buffer_size, self.batch_size, n_node_feature_missile, n_node_feature_enemy,
                                    self.action_size, n_step=self.n_step, per_alpha=per_alpha)
        self.n_node_feature_missile = n_node_feature_missile
        self.n_node_feature_enemy = n_node_feature_enemy

        self.action_space = [i for i in range(self.action_size)]
        self.iqn_N = iqn_N
        self.n_cos = n_cos
        self.GNN = GNN#
        # 추가추가
        self.gamma_n_step = torch.tensor([[self.gamma ** i for i in range(self.n_step + 1)] for _ in range(self.batch_size)], dtype=torch.float, device=device)

        self.node_representation_ship_feature = NodeEmbedding(feature_size=feature_size_ship,
                                                              n_representation_obs=n_representation_ship,
                                                              layers=node_embedding_layers_ship).to(device)  # 수정사항

        self.node_embedding = NodeEmbedding(feature_size=feature_size_missile,
                                            n_representation_obs=n_representation_action,
                                            layers=node_embedding_layers_missile).to(device)  # 수정사항

        self.node_representation_ship_feature_tar = NodeEmbedding(feature_size=feature_size_ship,
                                                                  n_representation_obs=n_representation_ship,
                                                                  layers=node_embedding_layers_ship).to(device)  # 수정사항

        self.node_embedding_tar = NodeEmbedding(feature_size=feature_size_missile,
                                                n_representation_obs=n_representation_action,
                                                layers=node_embedding_layers_missile).to(device)  # 수정사항

        self.DuelingQ = DuelingDQN().to(device)
        self.DuelingQtar = DuelingDQN().to(device)



        self.Q = DQNNetwork(state_action_size=n_representation_ship + n_representation_action,
                                  layers=layers).to(device)


        self.Q_tar = DQNNetwork(state_action_size=n_representation_ship + n_representation_action,
                                  layers=layers).to(device)


        self.node_representation_ship_feature_tar.load_state_dict(self.node_representation_ship_feature.state_dict())
        self.node_embedding_tar.load_state_dict(self.node_embedding.state_dict())
        self.Q_tar.load_state_dict(self.Q.state_dict())
        self.DuelingQtar.load_state_dict(self.DuelingQtar.state_dict())

        self.eval_params = list(self.DuelingQ.parameters()) + \
                           list(self.Q.parameters()) + \
                           list(self.node_embedding.parameters()) + \
                           list(self.node_representation_ship_feature.parameters())

        if cfg.optimizer == 'AdaHessian':
            self.optimizer = AdaHessian(self.eval_params, lr=learning_rate)
        if cfg.optimizer == 'LBFGS':
            self.optimizer = optim.LBFGS(self.eval_params, lr=learning_rate)
        if cfg.optimizer == 'ADAM':
            self.optimizer = optim.Adam(self.eval_params, lr=learning_rate)  #

        self.scheduler = StepLR(optimizer=self.optimizer, step_size=cfg.scheduler_step, gamma=cfg.scheduler_ratio)

        self.time_check = [[], []]

        self.dummy_node = [
            [[0] * feature_size_missile for _ in range(i)]
            for i in range(n_node_feature_missile)]

    def save_model(self, e, t, epsilon, path):

        torch.save({
            'e': e,
            't': t,
            'epsilon': epsilon,
            'Q': self.Q.state_dict(),
            'Q_tar': self.Q_tar.state_dict(),
            'node_representation_ship_feature': self.node_representation_ship_feature.state_dict(),
            'dueling_Q': self.DuelingQ.state_dict(),
            'optimizer': self.optimizer.state_dict()}, "{}".format(path))

    def eval_check(self, eval):
        if eval == True:
            self.DuelingQ.eval()
            self.DuelingQtar.eval()

            self.node_representation_ship_feature.eval()
            self.node_representation_ship_feature_tar.eval()

            self.node_embedding.eval()
            self.node_embedding_tar.eval()

            self.Q.eval()
            self.Q_tar.eval()
        else:
            self.DuelingQ.train()
            self.node_representation_ship_feature.train()
            self.node_embedding.train()
            self.Q.train()

    def load_model(self, path):
        checkpoint = torch.load(path)
        e = checkpoint["e"]
        t = checkpoint["t"]
        epsilon = checkpoint["epsilon"]
        self.Q.load_state_dict(checkpoint["Q"])
        self.Q_tar.load_state_dict(checkpoint["Q_tar"])
        self.node_representation_ship_feature.load_state_dict(checkpoint["node_representation_ship_feature"])
        self.DuelingQ.load_state_dict(checkpoint["dueling_Q"])
        return e, t, epsilon

    def get_node_representation(self, missile_node_feature,
                                ship_features,
                                edge_index_missile,
                                n_node_features_missile,
                                node_cats,
                                mini_batch=False,
                                target=False):

        if mini_batch == False:
            with torch.no_grad():
                ship_features = torch.tensor(ship_features, dtype=torch.float, device=device)
                node_embedding_ship_features = self.node_representation_ship_feature(ship_features)
                missile_node_feature = torch.tensor(missile_node_feature, dtype=torch.float,
                                                    device=device).clone().detach()
                node_representation = torch.cat([node_embedding_ship_features], dim=1)
                missile_node_feature = self.node_embedding(missile_node_feature)
                return node_representation, missile_node_feature
        else:

            """ship feature 만드는 부분"""
            ship_features = torch.tensor(ship_features, dtype=torch.float).to(device).squeeze(1)
            if target == False:
                node_embedding_ship_features = self.node_representation_ship_feature(ship_features)
            else:
                node_embedding_ship_features = self.node_representation_ship_feature_tar(ship_features)
            max_len = np.max([len(mnf) for mnf in missile_node_feature])
            if max_len <= self.action_size:
                max_len = self.action_size
            temp = list()
            for mnf in missile_node_feature:
                temp.append(
                    torch.cat([torch.tensor(mnf), torch.tensor(self.dummy_node[max_len - len(mnf)])], dim=0).tolist())
            missile_node_feature = torch.tensor(temp, dtype=torch.float).to(device)

            missile_node_size = missile_node_feature.shape[1]
            missile_node_feature = missile_node_feature.reshape(self.batch_size * missile_node_size, -1)
            if target == False:
                missile_node_feature = self.node_embedding(missile_node_feature)
            else:
                missile_node_feature = self.node_embedding_tar(missile_node_feature)
            missile_node_feature = missile_node_feature.reshape(self.batch_size, missile_node_size, -1)
            node_representation = torch.cat([node_embedding_ship_features], dim=1)
            return node_representation, missile_node_feature

    def cal_Q(self, obs, obs_graph, action_feature, action_features, avail_actions, agent_id, target=False,
              action_index=None):
        """
        node_representation
        - training 시        : batch_size X num_nodes X feature_size
        - action sampling 시 : num_nodes X feature_size
        """
        if target == False:
            mask = torch.tensor(avail_actions, device=device).bool()
            action_features = obs_graph[:, :self.action_size, :]
            action_index = action_index.unsqueeze(1).unsqueeze(2).expand(-1, -1, action_features.size(2))
            action_feature = torch.gather(action_features, 1, action_index).squeeze(1)
            obs_n_action = torch.cat([obs, action_feature], dim=1)


            obs_expand = obs.unsqueeze(1)
            obs_expand = obs_expand.expand(
                [self.batch_size, self.action_size, obs_expand.shape[2]])  # batch-size, action_size, obs_size

            obs_n_action = torch.cat([obs_expand, action_features], dim=2)
            obs_n_action_flat = obs_n_action.reshape(self.action_size * self.batch_size, obs_n_action.size(-1))


            Q = self.Q(obs_n_action_flat)
            Q = Q.view(obs_n_action.size(0), self.action_size)

            return Q
        else:
            with torch.no_grad():
                mask = torch.tensor(avail_actions, device=device).bool()
                action_features = obs_graph[:, :self.action_size, :]
                batch_size = action_features.shape[0]

                obs_expand = obs.unsqueeze(1)
                obs_expand = obs_expand.expand([self.batch_size, self.action_size, obs_expand.shape[2]])
                # print(obs_expand.shape, action_features.shape, self.action_size)
                obs_n_action = torch.cat([obs_expand, action_features], dim=2)
                # obs_n_action : Q(s,a) -> Q(s||a)

                obs_n_action_flat = obs_n_action.reshape(self.action_size * self.batch_size, obs_n_action.size(-1))


                Q_tar = self.Q_tar(obs_n_action_flat)

                Q_tar = Q_tar.view(obs_n_action.size(0), self.action_size)
                print(Q_tar.shape)
                action_max = Q_tar.max(dim=1)[1].long().unsqueeze(1)
                Q_tar_max = torch.gather(Q_tar, 1, action_max)
                return Q_tar_max

    @torch.no_grad()
    def sample_action(self, node_representation, node_representation_graph, avail_action, epsilon, action_feature,
                      training=True, with_noise=False, boltzman=False, step=None):
        """

        node_representation 차원 : n_agents X n_representation_comm
        action_feature 차원      : action_size X n_action_feature
        avail_action 차원        : n_agents X action_size
        """
        action_feature_dummy = action_feature
        node_embedding_action = node_representation_graph[0:self.action_size, :]
        obs_n_action = torch.cat(
            [node_representation.expand(node_embedding_action.shape[0], node_representation.shape[1]),
             node_embedding_action], dim=1)

        mask = torch.tensor(avail_action, device=device).bool()
        "Dueling Q값을 계산하는 부분"
        action_size = obs_n_action.shape[0]
        if action_size >= self.action_size:
            action_size = self.action_size
        Q = torch.stack([self.Q(obs_n_action[i].unsqueeze(0)) for i in
                         range(action_size)]).squeeze(1).squeeze(1).unsqueeze(0)
        remain_action = torch.tensor([float('-inf') for _ in range(self.action_size - action_size)],
                                     device=device).unsqueeze(0)
        Q = torch.cat([Q, remain_action], dim=1)
        Q = Q.masked_fill(mask == 0, float('-inf'))
        greedy_u = torch.argmax(Q)
        if np.random.uniform(0, 1) >= epsilon:
            u = greedy_u.detach().item()
        else:
            mask_n = np.array(avail_action[0], dtype=np.float64)
            u = np.random.choice(self.action_space, p=mask_n / np.sum(mask_n))
        action_blue = action_feature_dummy[u]
        return action_blue, u

    def learn(self):

        node_features_missile, \
        ship_features, \
        edge_indices_missile, \
        actions, \
        rewards, \
        dones, \
        node_features_missile_next, \
        ship_features_next, \
        edge_indices_missile_next, \
        avail_actions, \
        avail_actions_next, \
        status, \
        status_next, \
        priority, \
        batch_index, \
        action_feature, action_features, action_features_next, heterogenous_edges, heterogenous_edges_next, action_index, node_cats, node_cats_next = self.buffer.sample()


        action_index = torch.tensor(action_index, device=device, dtype=torch.long)

        """
        node_features : batch_size x num_nodes x feature_size
        actions : batch_size x num_agents
        action_feature :     batch_size x action_size x action_feature_size
        avail_actions_next : batch_size x num_agents x action_size 
        """
        n_node_features_missile = self.n_node_feature_missile
        dones = torch.tensor(dones, device=device, dtype=torch.float)
        rewards = torch.tensor(rewards, device=device, dtype=torch.float)


        self.eval_check(eval=True)
        obs_next, obs_next_graph = self.get_node_representation(
            node_features_missile_next,
            ship_features_next,
            heterogenous_edges_next,
            n_node_features_missile,
            node_cats=node_cats_next,
            mini_batch=True,
            target=True)
        q_tot_tar = self.cal_Q(obs=obs_next,
                               obs_graph=obs_next_graph,
                               action_feature=None,
                               action_features=action_features_next,
                               avail_actions=avail_actions_next,
                               agent_id=0,
                               target=True)
        self.eval_check(eval=False)


        obs, obs_graph = self.get_node_representation(
            node_features_missile,
            ship_features,
            heterogenous_edges,
            n_node_features_missile,
            node_cats=node_cats,
            mini_batch=True)
        q_tot = self.cal_Q(obs=obs,
                           obs_graph=obs_graph,
                           action_feature=action_feature,
                           action_features=action_features,
                           avail_actions=avail_actions,
                           agent_id=0,
                           target=False,
                           action_index=action_index)

        rewards_1_step = rewards[:, 0].unsqueeze(1)
        rewards_k_step = rewards[:, 1:]
        masked_n_step_bootstrapping = (1 - dones) * torch.cat([rewards_k_step, q_tot_tar], dim=1)
        discounted_n_step_bootstrapping = self.gamma_n_step * torch.cat([rewards_1_step, masked_n_step_bootstrapping],
                                                                        dim=1)
        td_target = discounted_n_step_bootstrapping.sum(dim=1, keepdims=True)

        loss = F.huber_loss(q_tot, td_target)


        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.eval_params, cfg.grad_clip)
        self.optimizer.step()
        tau = 5e-4
        for target_param, local_param in zip(self.Q_tar.parameters(), self.Q.parameters()):
            target_param.data.copy_(tau * local_param.data + (1 - tau) * target_param.data)
        for target_param, local_param in zip(self.DuelingQtar.parameters(), self.DuelingQ.parameters()):
            target_param.data.copy_(tau * local_param.data + (1 - tau) * target_param.data)

        for target_param, local_param in zip(self.node_embedding_tar.parameters(), self.node_embedding.parameters()):
            target_param.data.copy_(tau * local_param.data + (1 - tau) * target_param.data)
        for target_param, local_param in zip(self.node_representation_ship_feature_tar.parameters(),
                                             self.node_representation_ship_feature.parameters()):
            target_param.data.copy_(tau * local_param.data + (1 - tau) * target_param.data)