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

#import torch.cuda.OutOfMemoryError

def weight_init_xavier_uniform(submodule):
    if isinstance(submodule, torch.nn.Conv2d):
        torch.nn.init.xavier_uniform_(submodule.weight)
        submodule.bias.data.fill_(0.01)
    if isinstance(submodule, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(submodule.weight)
    elif isinstance(submodule, torch.nn.BatchNorm2d):
        submodule.weight.data.fill_(1.0)
        submodule.bias.data.zero_()

class IQN(nn.Module):
    def __init__(self, state_size_advantage, state_size_value, action_size, batch_size, layer_size=128, N=12, layers = [128, 96, 64, 56, 48], n_cos = 64):
        super(IQN, self).__init__()

        self.state_size_advantage = state_size_advantage
        self.state_size_value = state_size_value
        self.batch_size = batch_size
        self.action_size = action_size
        self.N = N
        self.n_cos = n_cos
        self.layer_size = layer_size
        self.pis = torch.FloatTensor([np.pi * i for i in range(self.n_cos)]).view(1, 1, self.n_cos).to(device)
        self.head_a = nn.Linear(self.state_size_advantage, layer_size)  # cound be a cnn
        self.head_v = nn.Linear( self.state_size_value, layer_size)  # cound be a cnn
        self.cos_embedding = nn.Linear(self.n_cos, layer_size)
        self.noisylinears_for_advantage = OrderedDict()
        last_layer = layer_size
        for i in range(len(layers)):
            layer = layers[i]
            if i <= len(layers)-2:
                if cfg.epsilon_greedy == False:
                    self.noisylinears_for_advantage['linear{}'.format(i)]= NoisyLinear(last_layer, layer)
                else:
                    self.noisylinears_for_advantage['linear{}'.format(i)] = nn.Linear(last_layer, layer)
                self.noisylinears_for_advantage['batchnorm{}'.format(i)] = nn.BatchNorm1d(layer)
                self.noisylinears_for_advantage['activation{}'.format(i)] = nn.ELU()
                last_layer = layer
            else:
                if cfg.epsilon_greedy == False:
                    self.noisylinears_for_advantage['linear{}'.format(i)] = NoisyLinear(last_layer,1)
                else:
                    self.noisylinears_for_advantage['linear{}'.format(i)] = nn.Linear(last_layer, 1)
        self.advantage_layer = nn.Sequential(self.noisylinears_for_advantage)
        if cfg.epsilon_greedy == False:
            self.reset_noise_net()
        else:
            self.advantage_layer.apply(weight_init_xavier_uniform)


    def reset_noise_net(self):
        for layer in self.advantage_layer:
            if type(layer) is NoisyLinear:
                layer.reset_noise()

    def remove_noise_net(self):
        for layer in self.advantage_layer:
            if type(layer) is NoisyLinear:
                layer.remove_noise()

    def calc_cos(self, batch_size):
        """
        Calculating the cosinus values depending on the number of tau samples
        """
        taus = torch.rand(batch_size, self.N).to(device).unsqueeze(-1)  # (batch_size, self.N, 1)
        #print("전", taus.shape, self.pis.shape)
        cos = torch.cos(taus * self.pis)  # self.pis shape : 1,1,self.n_cos
        #print("후", cos.shape)
        assert cos.shape == (batch_size, self.N, self.n_cos), "cos shape is incorrect"
        return cos, taus

    def advantage_forward(self, input, cos, mini_batch):
        N = self.N
        if mini_batch == False:
            batch_size = 1
        else:
            batch_size = input.shape[0]
        x = torch.relu(self.head_a(input.to(device)))  # x의 shape는 batch_size, layer_size

        cos = cos.view(batch_size * N, self.n_cos)
        cos_x = torch.relu(self.cos_embedding(cos)).view(batch_size, N, self.layer_size)  # (batch, n_tau, layer)

        x = (x.unsqueeze(1) * cos_x).view(batch_size * N, self.layer_size)  # 이부분이 phsi * phi에 해당하는 부분
        out_a = self.advantage_layer(x)
        quantiles_a = out_a.view(batch_size, N, 1)
        a = quantiles_a.mean(dim=1)
        return a

    def value_forward(self, input, cos, mini_batch):
        N = self.N
        if mini_batch == False:
            batch_size = 1
        else:
            batch_size = input.shape[0]
        y = torch.relu(self.head_v(input.to(device)))  # x의 shape는 batch_size, layer_size
        cos_y = cos.view(batch_size * N, self.n_cos)
        cos_y = torch.relu(self.cos_embedding(cos_y)).view(batch_size, N, self.layer_size)  # (batch, n_tau, layer)
        #print(y.unsqueeze(1).shape, cos_y.shape)
        y = (y.unsqueeze(1) * cos_y).view(batch_size * N, self.layer_size)  # 이부분이 phsi * phi에 해당하는 부분
        out_v = self.advantage_layer(y)
        quantiles_v = out_v.view(batch_size, N, 1)
        v = quantiles_v.mean(dim=1)
        return v


class DuelingDQN(nn.Module):
    def __init__(self):
        super(DuelingDQN, self).__init__()

    def forward(self, V, A, mask, past_action = None, training = False):
        # v의 shape : batch_size x 1
        # a의 shape : batch_size x action size
        if (past_action == None) and (training == False):
            A = A.masked_fill(mask == 0,float('-inf'))
            zeros = torch.zeros_like(A)
            ones = torch.ones_like(A)
            nA = torch.where(A == float('-inf'), zeros, ones).sum()
            mean_A = torch.where(A == float('-inf'), zeros, A).sum()
            mean_A = mean_A/nA
            Q = V+A-mean_A

        if (past_action != None) and (training == True):
            #print(A.shape, mask.shape)

            mask = mask.squeeze(1)
            A = A.masked_fill(mask == 0, float('-inf'))
            zeros = torch.zeros_like(A)
            ones = torch.ones_like(A)
            nA = torch.where(A == float('-inf'), zeros, ones).sum(dim = 1)
            mean_A = torch.where(A == float('-inf'), zeros, A).sum(dim = 1)
            mean_A = mean_A / nA
            Q = V + past_action - mean_A.unsqueeze(1)
        if (past_action == None) and (training == True):
            mask = mask.squeeze(1)
            A = A.masked_fill(mask == 0, float('-inf'))

            zeros = torch.zeros_like(A)
            ones = torch.ones_like(A)

            nA = torch.where(A == float('-inf'), zeros, ones).sum(dim = 1)
            mean_A = torch.where(A == float('-inf'), zeros, A).sum(dim = 1)

            mean_A = mean_A / nA
            Q = V + A - mean_A.unsqueeze(1)


        return Q

class VDN(nn.Module):

    def __init__(self):
        super(VDN, self).__init__()

    def forward(self, q_local):
        return torch.sum(q_local, dim=1)


class Network(nn.Module):
    def __init__(self, state_size, state_action_size, layers=[8,12]):
        super(Network, self).__init__()
        self.state_size = state_size
        self.state_action_size = state_action_size
        self.NN_sequential = OrderedDict()
        self.fc_pi = nn.Linear(state_action_size, layers[0])
        self.fc_v = nn.Linear(state_size, layers[0])
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
        self.output_pi = nn.Linear(last_layer, 1)
        self.output_v = nn.Linear(last_layer, 1)




    def pi(self, x, visualize = False):
        if visualize == False:
            x = self.fc_pi(x)
            x = F.elu(x)
            x = self.forward_cal(x)
            pi = self.output_pi(x)
            return pi
        else:
            x = self.fc_pi(x)
            x = F.elu(x)
            x = self.forward_cal(x)
            pi = self.output_pi(x)
            return x

    def v(self, x):
        x = self.fc_v(x)
        x = F.elu(x)
        x = self.forward_cal(x)
        v = self.output_v(x)
        return v


class NodeEmbedding(nn.Module):
    def __init__(self, feature_size, n_representation_obs, layers = [20, 30 ,40]):
        super(NodeEmbedding, self).__init__()
        self.feature_size = feature_size
        self.linears = OrderedDict()
        last_layer = self.feature_size
        for i in range(len(layers)):
            layer = layers[i]
            if i <= len(layers)-2:
                self.linears['linear{}'.format(i)]= nn.Linear(last_layer, layer)
                self.linears['activation{}'.format(i)] = nn.ELU()
                last_layer = layer
            else:
                self.linears['linear{}'.format(i)] = nn.Linear(last_layer, n_representation_obs)

        self.node_embedding = nn.Sequential(self.linears)
        #print(self.node_embedding)
        self.node_embedding.apply(weight_init_xavier_uniform)


    def forward(self, node_feature, missile=False):
        #print(node_feature.shape)
        node_representation = self.node_embedding(node_feature)
        return node_representation


class Replay_Buffer:
    def __init__(self, buffer_size, batch_size, n_node_feature_missile, n_node_feature_enemy, action_size, n_step, per_alpha):
        self.buffer = deque()
        self.alpha = per_alpha
        self.step_count_list = list()
        for _ in range(18):
            self.buffer.append(deque(maxlen=buffer_size))
        self.buffer_size = buffer_size
        self.n_node_feature_missile = n_node_feature_missile
        self.n_node_feature_enemy = n_node_feature_enemy
        #self.agent_id = np.eye(self.num_agent).tolist()
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

        #print(len(self.buffer[7]), len(self.buffer[10]), len(self.step_count_list))

#ship_features
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
                yield datas[7][s+self.n_step]
            if cat == 'status':
                yield datas[8][s]
            if cat == 'status_next':
                yield datas[8][s+self.n_step]

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
                yield datas[17][s+self.n_step]

    def update_transition_priority(self, batch_index, delta):

        copied_delta_store = deepcopy(list(self.buffer[10]))
        delta = np.abs(delta).reshape(-1) + np.min(copied_delta_store)
        priority = np.array(copied_delta_store).astype(float)

        batch_index = batch_index.astype(int)
        priority[batch_index] = delta
        self.buffer[10]= deque(priority, maxlen=self.buffer_size)


    def sample(self):
        step_count_list = self.step_count_list[:]

        priority = list(deepcopy(self.buffer[10]))[:-self.n_step]

        p = np.array(priority)**self.alpha
        p /= p.sum()
        p_sampled = p
        p = p.tolist()
        try:
            sampled_batch_idx = np.random.choice(step_count_list[:-self.n_step], size=self.batch_size, p = p)
        except ValueError as VE:
            sampled_batch_idx = np.random.choice(step_count_list[:-self.n_step], size=self.batch_size)
            print("value error 발생 중")

        p_sampled = p_sampled[sampled_batch_idx]


        node_feature_missile = self.generating_mini_batch(self.buffer, sampled_batch_idx, cat='node_feature_missile')
        node_features_missile = list(node_feature_missile)

        edge_index_missile = None#self.generating_mini_batch(self.buffer, sampled_batch_idx, cat='edge_index_missile')
        edge_indices_missile = None#list(edge_index_missile)

        node_feature_missile_next = self.generating_mini_batch(self.buffer, sampled_batch_idx, cat='node_feature_missile_next')
        node_features_missile_next = list(node_feature_missile_next)

        edge_index_missile_next = None#self.generating_mini_batch(self.buffer, sampled_batch_idx, cat='edge_index_missile_next')
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

        heterogenous_edges_next = self.generating_mini_batch(self.buffer, sampled_batch_idx, cat='heterogeneous_edges_next')
        heterogenous_edges_next = list(heterogenous_edges_next)

        action_index = self.generating_mini_batch(self.buffer, sampled_batch_idx,
                                                             cat='action_index')
        action_index = list(action_index)

        node_cats= self.generating_mini_batch(self.buffer, sampled_batch_idx,
                                                             cat='node_cats')
        node_cats = list(node_cats)

        node_cats_next= self.generating_mini_batch(self.buffer, sampled_batch_idx,
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
               avail_actions,\
               avail_actions_next, \
               status, \
               status_next,\
               priority, \
               sampled_batch_idx, \
               p_sampled,\
               action_feature, \
               action_features, \
               action_features_next,heterogenous_edges,heterogenous_edges_next, action_index, node_cats, node_cats_next





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
                 num_nodes

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

        self.buffer = Replay_Buffer(self.buffer_size, self.batch_size, n_node_feature_missile,n_node_feature_enemy, self.action_size, n_step = self.n_step, per_alpha = per_alpha)
        self.n_node_feature_missile = n_node_feature_missile
        self.n_node_feature_enemy = n_node_feature_enemy

        self.action_space = [i for i in range(self.action_size)]
        self.iqn_N = iqn_N
        self.n_cos=n_cos
        self.GNN = GNN

        self.gamma_n_step = torch.tensor([[self.gamma**i for i in range(self.n_step+1)] for _ in range(self.batch_size)], dtype = torch.float, device = device)


        self.node_representation_ship_feature = NodeEmbedding(feature_size=feature_size_ship,
                                                         n_representation_obs=n_representation_ship,
                                                         layers = node_embedding_layers_ship).to(device)  # 수정사항

        self.node_embedding = NodeEmbedding(feature_size=feature_size_missile,
                                                              n_representation_obs=n_representation_action,
                                                              layers=node_embedding_layers_missile).to(device)  # 수정사항




        self.Q = IQN(state_size_advantage=n_representation_ship + n_representation_action,
                     state_size_value=n_representation_ship,
                     action_size=self.action_size,
                     batch_size=self.batch_size, layer_size=iqn_layer_size, N=iqn_N, n_cos=n_cos,
                     layers=iqn_layers).to(device)

        self.Q_tar = IQN(
            state_size_advantage=n_representation_ship + n_representation_action,
            state_size_value=n_representation_ship,
            action_size=self.action_size,
            batch_size=self.batch_size, layer_size=iqn_layer_size, N=iqn_N, n_cos=n_cos, layers=iqn_layers).to(
            device)

        self.Q_tar.load_state_dict(self.Q.state_dict())
        self.DuelingQtar.load_state_dict(self.DuelingQtar.state_dict())

        self.eval_params = list(self.DuelingQ.parameters()) + \
                           list(self.Q.parameters()) + \
                           list(self.node_embedding.parameters()) + \
                           list(self.node_representation_ship_feature.parameters())






        if cfg.optimizer == 'AdaHessian':
            self.optimizer =AdaHessian(self.eval_params, lr=learning_rate)
        if cfg.optimizer == 'LBFGS':
            self.optimizer = optim.LBFGS(self.eval_params, lr=learning_rate)
        if cfg.optimizer == 'ADAM':
            self.optimizer = optim.Adam(self.eval_params, lr=learning_rate) #

        self.scheduler = StepLR(optimizer=self.optimizer, step_size=cfg.scheduler_step, gamma=cfg.scheduler_ratio)

        self.time_check = [[], []]

        self.dummy_node = [
                          [[0]*feature_size_missile for _ in range(i)]
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
            self.node_representation_ship_feature.eval()

            self.Q.eval()
            self.Q_tar.eval()

        else:
            self.DuelingQ.train()
            self.node_representation_ship_feature.train()

            self.Q.train()
            self.Q_tar.train()

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
                                mini_batch=False):

        if mini_batch == False:
            with torch.no_grad():
                ship_features = torch.tensor(ship_features, dtype=torch.float, device=device)
                node_embedding_ship_features = self.node_representation_ship_feature(ship_features)
                missile_node_feature = torch.tensor(missile_node_feature, dtype=torch.float,device=device).clone().detach()
                node_representation = torch.cat([node_embedding_ship_features], dim=1)
                missile_node_feature = self.node_embedding(missile_node_feature)
                return node_representation, missile_node_feature
        else:
            """ship feature 만드는 부분"""
            ship_features = torch.tensor(ship_features,dtype=torch.float).to(device).squeeze(1)
            node_embedding_ship_features = self.node_representation_ship_feature(ship_features)
            max_len = np.max([len(mnf) for mnf in missile_node_feature])
            if max_len <= self.action_size:
                max_len = self.action_size
            temp = list()
            for mnf in missile_node_feature:
                temp.append(torch.cat([torch.tensor(mnf), torch.tensor(self.dummy_node[max_len - len(mnf)])], dim=0).tolist())
            missile_node_feature = torch.tensor(temp, dtype=torch.float).to(device)

            missile_node_size = missile_node_feature.shape[1]
            missile_node_feature = missile_node_feature.reshape(self.batch_size*missile_node_size, -1)
            missile_node_feature = self.node_embedding(missile_node_feature)
            missile_node_feature = missile_node_feature.reshape(self.batch_size, missile_node_size, -1)
            node_representation = torch.cat([node_embedding_ship_features], dim=1)
            return node_representation, missile_node_feature





    def cal_Q(self, obs, obs_graph, action_feature, action_features, avail_actions, agent_id, cos, target=False, action_index = None):
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
            obs_n_action = torch.cat([obs, action_feature], dim = 1)
            A_a = self.Q.advantage_forward(obs_n_action, cos, mini_batch=True)

            obs_expand = obs.unsqueeze(1)
            obs_expand = obs_expand.expand([self.batch_size, self.action_size, obs_expand.shape[2]])  # batch-size, action_size, obs_size

            obs_n_action = torch.cat([obs_expand, action_features], dim=2)
            obs_n_action_flat = obs_n_action.reshape(self.action_size*self.batch_size, obs_n_action.size(-1))


            cos1 = cos.expand([self.action_size, self.batch_size, self.iqn_N, self.n_cos])
            cos1 = cos1.reshape(self.action_size*self.batch_size, self.iqn_N, self.n_cos)
            q_values_flat = self.Q.advantage_forward(obs_n_action_flat, cos1, mini_batch=True)
            q_values = q_values_flat.view(obs_n_action.size(0), self.action_size)
            A = torch.stack([q_values[:, i] for i in range(self.action_size)], dim=1)
            V = self.Q.value_forward(obs, cos, mini_batch = True)



            Q = self.DuelingQ(V, A, mask, past_action = A_a, training = True)
            # del mask, action_features, action_index, action_feature, A_a, obs_expand, obs_n_action, obs_n_action_flat, cos1, q_values_flat, q_values, A, V
            # torch.cuda.empty_cache()
            return Q
        else:
            with torch.no_grad():
                mask = torch.tensor(avail_actions, device=device).bool()
                action_features = obs_graph[:, :self.action_size, :]
                batch_size = action_features.shape[0]

                obs_expand = obs.unsqueeze(1)
                obs_expand = obs_expand.expand([self.batch_size, self.action_size, obs_expand.shape[2]])
                #print(obs_expand.shape, action_features.shape, self.action_size)
                obs_n_action = torch.cat([obs_expand, action_features], dim=2)
                # obs_n_action : Q(s,a) -> Q(s||a)

                obs_n_action_flat = obs_n_action.reshape(self.action_size * self.batch_size, obs_n_action.size(-1))

                cos1 = cos.expand([self.action_size, self.batch_size, self.iqn_N, self.n_cos])
                cos1 = cos1.reshape(self.action_size * self.batch_size, self.iqn_N, self.n_cos)
                q_values_flat = self.Q.advantage_forward(obs_n_action_flat, cos1, mini_batch=True)
                q_values = q_values_flat.view(obs_n_action.size(0), self.action_size)



                A = torch.stack([q_values[:, i] for i in range(self.action_size)], dim=1)
                V = self.Q.value_forward(obs, cos, mini_batch=True)
                Q = self.DuelingQ(V, A, mask, past_action=None, training=True)


                q_values_flat = self.Q_tar.advantage_forward(obs_n_action_flat, cos1, mini_batch=True)
                q_values = q_values_flat.view(obs_n_action.size(0), self.action_size)
                A_tar = torch.stack([q_values[:, i] for i in range(self.action_size)], dim=1)
                V_tar = self.Q_tar.value_forward(obs,
                                                 cos,
                                                 mini_batch = True)
                Q_tar = self.DuelingQtar(V_tar,
                                         A_tar,
                                         mask, past_action=None, training=True)
                action_max = Q.max(dim = 1)[1].long().unsqueeze(1)
                Q_tar_max = torch.gather(Q_tar, 1, action_max)
                return Q_tar_max


    @torch.no_grad()
    def sample_action(self, node_representation, node_representation_graph, avail_action, epsilon, action_feature, training = True, with_noise = False, boltzman = False, step = None):
        """

        node_representation 차원 : n_agents X n_representation_comm
        action_feature 차원      : action_size X n_action_feature
        avail_action 차원        : n_agents X action_size
        """
        action_feature_dummy = action_feature
        node_embedding_action = node_representation_graph[0:self.action_size, :]
        obs_n_action = torch.cat([node_representation.expand(node_embedding_action.shape[0],node_representation.shape[1]),node_embedding_action], dim = 1)

        mask = torch.tensor(avail_action, device=device).bool()
        "Dueling Q값을 계산하는 부분"
        cos, taus = self.Q.calc_cos(1)
        V = self.Q.value_forward(node_representation, cos, mini_batch=False)
        action_size = obs_n_action.shape[0]
        if action_size >= self.action_size:
             action_size = self.action_size
        A = torch.stack([self.Q.advantage_forward(obs_n_action[i].unsqueeze(0), cos, mini_batch=False) for i in range(action_size)]).squeeze(1).squeeze(1).unsqueeze(0)
        remain_action = torch.tensor([float('-inf') for _ in range(self.action_size-action_size)], device = device).unsqueeze(0)
        A = torch.cat([A, remain_action], dim = 1)
        Q = self.DuelingQ(V, A, mask)
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
        status_next,\
        priority,\
        batch_index, \
        p_sampled, action_feature, action_features, action_features_next, heterogenous_edges,heterogenous_edges_next,action_index, node_cats, node_cats_next = self.buffer.sample()





        weight = ((len(self.buffer.buffer[10])-self.n_step)*torch.tensor(priority, dtype=torch.float, device = device))**(-self.beta)
        action_index = torch.tensor(action_index, device = device, dtype = torch.long)
        weight /= weight.max()
        """
        node_features : batch_size x num_nodes x feature_size
        actions : batch_size x num_agents
        action_feature :     batch_size x action_size x action_feature_size
        avail_actions_next : batch_size x num_agents x action_size 
        """
        n_node_features_missile = self.n_node_feature_missile
        dones = torch.tensor(dones, device=device, dtype=torch.float)
        rewards = torch.tensor(rewards, device=device, dtype=torch.float)
        cos, taus = self.Q.calc_cos(self.batch_size)


        self.eval_check(eval=True)
        obs_next, obs_next_graph = self.get_node_representation(
            node_features_missile_next,
            ship_features_next,
            heterogenous_edges_next,
            n_node_features_missile,
            node_cats = node_cats_next,
            mini_batch=True)
        q_tot_tar = self.cal_Q(obs=obs_next,
                           obs_graph = obs_next_graph,
                        action_feature=None,
                        action_features=action_features_next,
                        avail_actions=avail_actions_next,
                        agent_id=0,
                        target=True,
                        cos=cos)
        self.eval_check(eval = False)

        cos, taus = self.Q.calc_cos(self.batch_size)
        obs, obs_graph = self.get_node_representation(
            node_features_missile,
            ship_features,
            heterogenous_edges,
            n_node_features_missile,
            node_cats=node_cats,
            mini_batch=True)
        q_tot = self.cal_Q(obs=obs,
                       obs_graph = obs_graph,
                       action_feature=action_feature,
                       action_features=action_features,
                       avail_actions=avail_actions,
                       agent_id=0,
                       target=False,
                       cos=cos,  action_index = action_index)


        rewards_1_step = rewards[:, 0].unsqueeze(1)
        rewards_k_step = rewards[:, 1:]
        masked_n_step_bootstrapping = (1-dones)*torch.cat([rewards_k_step, q_tot_tar], dim = 1)
        discounted_n_step_bootstrapping = self.gamma_n_step*torch.cat([rewards_1_step, masked_n_step_bootstrapping], dim = 1)
        td_target = discounted_n_step_bootstrapping.sum(dim=1, keepdims = True)
        delta = (td_target - q_tot).detach().tolist()
        self.buffer.update_transition_priority(batch_index = batch_index, delta = delta)
        loss = F.huber_loss(weight * q_tot, weight * td_target)
        del node_features_missile, \
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
            p_sampled, action_feature, action_features, action_features_next, heterogenous_edges, heterogenous_edges_next, \
            action_index,cos, taus, q_tot, q_tot_tar, rewards_1_step, rewards_k_step,masked_n_step_bootstrapping, discounted_n_step_bootstrapping, td_target

        gc.collect()
        torch.cuda.empty_cache()

        self.optimizer.zero_grad()
        if type(self.optimizer) == AdaHessian:
            loss.backward(create_graph=True)
        else:
            loss.backward()
        gc.collect()
        torch.cuda.empty_cache()
        torch.nn.utils.clip_grad_norm_(self.eval_params, cfg.grad_clip)

        self.optimizer.step()
        self.scheduler.step()
        tau = 5e-4
        for target_param, local_param in zip(self.Q_tar.parameters(), self.Q.parameters()):
            target_param.data.copy_(tau * local_param.data + (1 - tau) * target_param.data)
        for target_param, local_param in zip(self.DuelingQtar.parameters(), self.DuelingQ.parameters()):
            target_param.data.copy_(tau * local_param.data + (1 - tau) * target_param.data)