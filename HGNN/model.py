import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict
import sys
sys.path.append("..")  # 상위 폴더를 import할 수 있도록 경로 추가
from cfg import get_cfg
from GTN.inits import glorot
cfg = get_cfg()
print(torch.cuda.device_count())
device =torch.device(cfg.cuda if torch.cuda.is_available() else "cpu")
print(device)

def weight_init_xavier_uniform(submodule):
    if isinstance(submodule, torch.nn.Conv2d):
        torch.nn.init.xavier_uniform_(submodule.weight)
        submodule.bias.data.fill_(0.01)
    if isinstance(submodule, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(submodule.weight)
    elif isinstance(submodule, torch.nn.BatchNorm2d):
        submodule.weight.data.fill_(1.0)
        submodule.bias.data.zero_()

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
                self.linears['batchnorm{}'.format(i)] = nn.BatchNorm1d(layer)
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

class HGNN(nn.Module):
    def __init__(self, feature_size, embedding_size, graph_embedding_size, layers, num_node_cat, num_edge_cat):
        super(HGNN, self).__init__()
        self.num_edge_cat = num_edge_cat
        self.graph_embedding_size = graph_embedding_size
        self.embedding_size = embedding_size
        self.Ws = []
        for i in range(num_edge_cat):
            self.Ws.append(nn.Parameter(torch.Tensor(feature_size, graph_embedding_size)))
        self.Ws = nn.ParameterList(self.Ws)
        [glorot(W) for W in self.Ws]
        self.embedding_layers = NodeEmbedding(graph_embedding_size*num_edge_cat, embedding_size, layers).to(device)

    #def forward(self, A, X, num_nodes=None, mini_batch=False):

    def forward(self, A, X, mini_batch, layer = 0):
        if mini_batch == False:
            temp = list()
            for e in range(len(A)):
                E = A[e]
                num_nodes = X.shape[0]
                E = torch.sparse_coo_tensor(E, torch.ones(torch.tensor(E).shape[1]), (num_nodes, num_nodes)).to(device).to_dense()
                H = E@X@self.Ws[e]
                temp.append(H)
            H = torch.cat(temp, dim = 1)
            #print(H.shape)
            H = self.embedding_layers(H)
            return H
        else:
            batch_size = X.shape[0]
            num_nodes = X.shape[1]
            #mat_a = [torch.zeros(self.num_edge_cat, num_nodes, num_nodes).to(device) for _ in range(batch_size)]
            empty = torch.zeros(batch_size, num_nodes, self.num_edge_cat, self.graph_embedding_size).to(device)

            for b in range(batch_size):
                for e in range(self.num_edge_cat):
                    E = torch.sparse_coo_tensor(A[b][e],
                                            torch.ones(torch.tensor(torch.tensor(A[b][e]).shape[1])),
                                            (num_nodes, num_nodes)).to(device).to_dense()
                    H = E@X[b]@self.Ws[e]
                    empty[b, :, e, :].copy_(H)

            H = empty.reshape(batch_size, num_nodes, self.num_edge_cat*self.graph_embedding_size)
            H = H.reshape(batch_size*num_nodes, self.num_edge_cat*self.graph_embedding_size)
            H = self.embedding_layers(H)
            H = H.reshape(batch_size, num_nodes, self.embedding_size)
            return H


            #Hs = torch.einsum('bcij, bcjk-> bcik', torch.einsum('bijk, ci  -> bcjk', mat_a, filter), H_)




