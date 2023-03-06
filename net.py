import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import math
from gnn import *

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = np.full((capacity),None)
        self._capacity = capacity
        self._position = 0
        self._buffer_size = 0 
    
    def push(self, transitions):
        """ transition: (states, previous_reps, joint_actions, rewards, next_states, next_previous_reps, dones, etc, durations, next_durations)
        """
        for trans in zip(*transitions):
            self.buffer[self._position] = trans
            self._position = (self._position+1) % self._capacity
            self._buffer_size = min(self._buffer_size+1,self._capacity)
    
    def sample(self, batch_size):
        idxs = np.random.choice(self._buffer_size, size=batch_size, replace=False)
        batch_trans = self.buffer[idxs]
        transitions = map(list, zip(*batch_trans))
        return transitions
    
    def clear(self):
        self.buffer = np.full((self.capacity),None)
        self._position = 0
        self._buffer_size = 0 

    def __len__(self):
        return self._buffer_size

TorchFloat = None
TorchLong = None

class Graphpool(nn.Module):
    def __init__(self, env, args, adjs, cs_tuple):
        """Initialization."""
        super(Graphpool, self).__init__()
        global TorchFloat,TorchLong
        TorchFloat = torch.cuda.FloatTensor if args.device == torch.device('cuda') else torch.FloatTensor
        TorchLong = torch.cuda.LongTensor if args.device == torch.device('cuda') else torch.LongTensor
        dpcs_mark, dpcs_id, spcs_id = cs_tuple
        self.adj_comp_list, self.adj_coop_list, self.grid2adj_comp, self.grid2adj_coop, self.distance = adjs
        self.adj_comp_dist = self.adj_comp_list.repeat(args.batch_size,1,1)
        self.adj_coop_dist = self.adj_coop_list.repeat(args.batch_size,1,1)
        self.env = env
        self.device = args.device
        self.topk = args.topk
        self.N = args.N
        self.topk_dpcs = int(len(dpcs_id)*args.k_h + 0.5)
        self.topk_spcs = int(len(spcs_id)*args.k_h + 0.5)
        self.time_dim = 4
        self.time_emb = nn.Embedding(args.T_LEN, self.time_dim, _weight=torch.rand(args.T_LEN,self.time_dim)) # Uniform(0,1)
        self.tp_dim = 2
        self.tp_emb = nn.Embedding(args.n_operator, self.tp_dim, _weight=torch.rand(args.n_operator,self.tp_dim)) # Uniform(0,1)
        self.cs_dim = 4
        self.cs_emb = nn.Embedding(args.N, self.cs_dim, _weight=torch.rand(args.N,self.cs_dim)) # Uniform(0,1)
        n_value_feat = 9
        self.com_dim = args.com_dim
        self.obs_dim = self.time_dim+self.tp_dim+self.cs_dim+args.com_dim*2+n_value_feat
        self.temp = args.temp
        
        self.c_weight = nn.Linear(args.hiddim*4, args.hiddim*4, bias=False)
        self.w_p = nn.Linear(self.obs_dim, args.hiddim)
        self.v_p_f = nn.Linear(args.hiddim, 1, bias=False)
        self.v_p_d = nn.Linear(args.hiddim, 1, bias=False)
    
        self.competition_agents = GAT(args, in_feat=n_value_feat+self.tp_dim+self.cs_dim, nhid=args.com_dim, dropout=args.dropout)
        self.cooperation_agents = GAT(args, in_feat=n_value_feat+self.tp_dim+self.cs_dim+args.com_dim, nhid=args.com_dim, dropout=args.dropout)
        self.neg_inf = -1e8*torch.ones(1,).to(args.device)

    def adj_to_dgledge(self, adj_station_full, adj_request_full, distance_full, inds, device=torch.cuda):
        B, _, _ = adj_station_full.shape
        adj_station, adj_request, distance = [], [], []
        if(inds is not None):
            for i in range(B):
                ind = inds[i].squeeze()
                adj_station.append(adj_station_full[i,ind][:,ind])
                adj_request.append(adj_request_full[i,ind][:,ind])
                distance.append(distance_full[ind][:,ind])
            adj_station = torch.stack(adj_station,dim=0) # (B,topk,topk)
            adj_request = torch.stack(adj_request,dim=0) # (B,topk,topk)
            distance = torch.stack(distance,dim=0) # (B,topk,topk)
        else:
            adj_station = adj_station_full
            adj_request = adj_request_full
            distance = distance_full.repeat(B,1,1)

        B, N, _ = adj_station.shape
        ### station-centric view
        edges_station = torch.nonzero(adj_station).T # (3, E)
        distance_feat = distance[edges_station[0], edges_station[1], edges_station[2]].unsqueeze(dim=0) # (1, E)
        view_feat = torch.zeros_like(distance_feat).repeat(2,1) # (2, E)
        view_feat[0] = 1
        feats_station = torch.cat([distance_feat, view_feat], dim=0)

        ### request-centirc viwe
        edges_request = torch.nonzero(adj_request).T # (3, E)
        distance_feat = distance[edges_request[0], edges_request[1], edges_request[2]].unsqueeze(dim=0) # (1, E)
        view_feat = torch.zeros_like(distance_feat).repeat(2,1) # (2, E)
        view_feat[1] = 1
        feats_request = torch.cat([distance_feat, view_feat], dim=0)

        edges = torch.cat([edges_station, edges_request], dim=-1)
        features = torch.cat([feats_station, feats_request], dim=-1).T # (E, 3)
        edges = edges[1:] + edges[0]*N # (2, E)
        edges = dgl.graph((edges[0], edges[1]), num_nodes=N*B, device=device)
        edges.edata['feature']=features.to(device)
        return edges

    def forward(self, state_action, query_grids, dpcs_mark, inds=None):
        """ 
        args: 
            state_action (B, topk, F)
            query_grids (B, 1)
            dpcs_mark (B, topk, 1)
            inds (B, topk, 1)
        """
        B, topk, F = state_action.shape
        time_emb = self.time_emb(state_action[...,0].type(TorchLong))
        tp_emb = self.tp_emb(state_action[...,1].type(TorchLong))
        cs_emb = self.cs_emb(self.env.cs_idx.unsqueeze(0)).repeat(B,1,1)
        if(inds is not None):
            cs_emb, _ = self.env.select_topk_cs(cs_emb, inds=inds)
        state_action_emb = torch.cat([cs_emb, tp_emb, state_action[...,2:].detach()],dim=-1)
        state_action_grad = torch.cat([cs_emb, tp_emb, state_action[...,2:]],dim=-1)

        adj_comp_req = torch.cat([self.grid2adj_comp[grid_idx] for grid_idx in query_grids],dim=0)
        dgl_edges_comp = self.adj_to_dgledge(self.adj_comp_dist, adj_comp_req, self.distance, inds, self.device)
        x = state_action_emb
        comp_reps = self.competition_agents(x, x, dgl_edges_comp)

        adj_coop_req = torch.cat([self.grid2adj_coop[grid_idx] for grid_idx in query_grids],dim=0)
        dgl_edges_coop = self.adj_to_dgledge(self.adj_coop_dist, adj_coop_req, self.distance, inds, self.device)
        x_m = torch.cat([state_action_emb, comp_reps],dim=-1)
        coop_reps = self.cooperation_agents(x_m, x_m, dgl_edges_coop)

        state_action_gnn = torch.cat([time_emb, state_action_grad, comp_reps, coop_reps],dim=-1)
        x_p = self.w_p(state_action_gnn)
        
        ### Dynamic pricing agents ###
        scores_d = self.v_p_d(x_p) # (B,topk,1)
        scores_d = scores_d - scores_d.max(dim=1,keepdim=True).values # for softmax stability
        scores_d = torch.where(dpcs_mark>1e-8,scores_d,self.neg_inf)
        val_topk, ind_topk = scores_d.topk(k=self.topk_dpcs,dim=-2) # (B,k,1)
        min_topk = val_topk.min(dim=-2,keepdim=True).values # (B,1,1)
        scores_d = torch.where(scores_d<min_topk,self.neg_inf,scores_d)
        weights_d = torch.softmax(scores_d/self.temp, dim=-2) # (B,topk,1)
        x_gate_d = x_p * weights_d # (B,topk,F')
        s_d = torch.cat([x_gate_d.sum(dim=-2), x_gate_d.max(dim=-2).values], dim=-1) # (B,F')

        ### Fixed pricing agents ###
        scores_f = self.v_p_f(x_p) # (B,topk,1)
        scores_f = scores_f - scores_f.max(dim=1,keepdim=True).values # for softmax stability
        scores_f = torch.where(dpcs_mark<1e-8,scores_f,self.neg_inf)
        val_topk, ind_topk = scores_f.topk(k=self.topk_spcs,dim=-2) # (B,k,1)
        min_topk = val_topk.min(dim=-2,keepdim=True).values # (B,1,1)
        scores_f = torch.where(scores_f<min_topk,self.neg_inf,scores_f)
        weights_f = torch.softmax(scores_f/self.temp, dim=-2) # (B,topk,1)
        x_gate_f = x_p * weights_f # (B,topk,1)
        s_f = torch.cat([x_gate_f.sum(dim=-2), x_gate_f.max(dim=-2).values], dim=-1) # (B,F')

        market_rep = torch.cat([s_f, s_d],dim=-1)
        return market_rep

class Critic(nn.Module):
    def __init__(self, env, args):
        """Initialization."""
        super(Critic, self).__init__()
        self.critic_net = nn.Sequential(
            nn.Linear(args.hiddim*4, args.hiddim),
            nn.ReLU(inplace=True),
            nn.Linear(args.hiddim, args.hiddim),
            nn.ReLU(inplace=True), 
            nn.Linear(args.hiddim, 1)
        )

    def forward(self, market_rep):
        q_values = self.critic_net(market_rep)
        # if(np.random.random() > 0.999):
        #     print('critic',q_values.max(),q_values.mean(),q_values.min())
        return q_values

class Hypernetworks(nn.Module):
    """ Multilayer perceptron. """
    def __init__(self, hiddens):
        """
        args:
            hiddens: A list of layers' dimensions in Hypernetworks.
        """
        super(Hypernetworks, self).__init__()
        activation_func = nn.Tanh()
        self.mlp_generator = nn.Sequential()
        n_layer = len(hiddens)
        for i in range(n_layer-1):
            self.mlp_generator.add_module('metafc_{}'.format(i), nn.Linear(hiddens[i],hiddens[i+1]))
            if(i < n_layer - 2):
                self.mlp_generator.add_module('actfunc_{}'.format(i), activation_func)
                
    def forward(self, meta_feats):
        ### output parameters ###
        parameters = self.mlp_generator(meta_feats)
        return parameters

class MetaGenerator(nn.Module):
    """ The MetaGenerator. """
    def __init__(self, in_dim, out_dim, meta_hiddens):
        """
        args:
            in_dim: Dimension of the input.
            out_dim: Dimension of the input.
            meta_hiddens: A list of hidden dimensions of Hypernetworks.
        """
        super(MetaGenerator, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.w_generator = Hypernetworks(meta_hiddens + [in_dim*out_dim,])
        self.b_generator = Hypernetworks(meta_hiddens + [1,])

    def forward(self, meta_feats, data):
        B,N,_ = meta_feats.shape
        meta_feats = meta_feats.view(N*B,1,-1) # (N*B,1,D1)
        data = data.view(N*B,1,-1) # (N*B,1,in_dim)
        ### policy generation ###
        weight = self.w_generator(meta_feats).view(-1, self.in_dim, self.out_dim) # (N*B, in_dim, out_dim)
        bias = self.b_generator(meta_feats).view(-1, 1, 1) # (N*B, 1, 1)
        output = torch.bmm(data, weight) + bias # (N*B,1,out_dim)
        return output.view(B,N,-1)

class Actor(nn.Module):
    def __init__(self, env, args, adjs):
        """Initialization."""
        super(Actor, self).__init__()
        self.device = args.device
        self.adj_comp_list, self.adj_coop_list, self.grid2adj_comp, self.grid2adj_coop, self.distance = adjs
        self.env = env
        self.N = args.N
        self.time_dim = 4
        self.time_emb = nn.Embedding(args.T_LEN, self.time_dim, _weight=torch.rand(args.T_LEN,self.time_dim)) # Uniform(0,1)
        self.cs_dim = 4
        self.cs_emb = nn.Embedding(args.N, self.cs_dim, _weight=torch.rand(args.N,self.cs_dim)) # Uniform(0,1)
        self.tp_dim = 2
        self.tp_emb = nn.Embedding(args.N, self.tp_dim, _weight=torch.rand(args.N,self.tp_dim)) # Uniform(0,1)
        self.com_dim = args.com_dim
        n_value_feat = 9 + self.cs_dim + self.tp_dim
        self.competition_agents = GAT(args, in_feat=n_value_feat, nhid=args.com_dim, graphtype="competition")
        self.cooperation_agents = GAT(args, in_feat=n_value_feat-1+args.com_dim, nhid=args.com_dim, graphtype="cooperation")
        self.obs_dim = self.time_dim + 2*args.com_dim + n_value_feat-1
        self.actor_net = nn.Sequential(
            nn.Linear(self.obs_dim, args.hiddim), 
            nn.ReLU(inplace=True),
            nn.Linear(args.hiddim, args.hiddim), # init uniform_(-stdv, stdv)
            nn.ReLU(inplace=True),
        )
        self.gru_encoder = nn.GRUCell(self.obs_dim+1, args.hiddim, bias=True)
        self.meta_generator = MetaGenerator(in_dim=args.hiddim, out_dim=1, meta_hiddens=[args.hiddim,32,16])
        self.zeros_tensor = torch.zeros((1,args.N,self.com_dim)).type(TorchFloat)

    def adj_to_dgledge(self, adj_station, adj_request, distance, device=torch.cuda):
        """
        adj_station (B, N, N)
        adj_request (B, N, N)
        distance (N, N)
        """
        ### station-centric view
        B, N, _ = adj_station.shape
        edges_station = torch.nonzero(adj_station).T # (3, E)
        distance_feat = distance[edges_station[1], edges_station[2]].unsqueeze(dim=0) # (1, E)
        view_feat = torch.zeros_like(distance_feat).repeat(2,1) # (2, E)
        view_feat[0] = 1
        feats_station = torch.cat([distance_feat, view_feat], dim=0) # (3, E)

        ### request-centirc viwe
        edges_request = torch.nonzero(adj_request).T # (3, E)
        distance_feat = distance[edges_request[1], edges_request[2]].unsqueeze(dim=0) # (1, E)
        view_feat = torch.zeros_like(distance_feat).repeat(2,1) # (2, E)
        view_feat[1] = 1
        feats_request = torch.cat([distance_feat, view_feat], dim=0) # (3, E)

        edges = torch.cat([edges_station, edges_request], dim=-1)
        features = torch.cat([feats_station, feats_request], dim=-1).T # (E, 3)
        edges = edges[1:] + edges[0]*N # (2, E)
        edges = dgl.graph((edges[0], edges[1]), num_nodes=N*B, device=device)
        edges.edata['feature']=features.to(device)
        return edges

    def forward(self, observation, query_grids, previous_reps):
        """
        observation: t_step,operator,supply_cp,demand,power,travel_count,duration,distance,supply_rest,electricfee,chargefee
        """
        B,N,_ = observation.shape
        h_pre, action_pre = previous_reps
        time_emb = self.time_emb(observation[...,0].type(TorchLong))
        tp_emb = self.tp_emb(observation[...,1].type(TorchLong))
        cs_emb = self.cs_emb(self.env.cs_idx.unsqueeze(0)).repeat(B,1,1)
        observe = torch.cat([cs_emb, tp_emb, observation[...,2:]], dim=-1)
        
        ### competition modeling ###
        adj_comp_dist = self.adj_comp_list.repeat(B,1,1) # (B, N, N)
        adj_comp_req = torch.cat([self.grid2adj_comp[grid_idx] for grid_idx in query_grids],dim=0) # (B, N, N)
        dgl_edges_comp = self.adj_to_dgledge(adj_comp_dist, adj_comp_req, self.distance, self.device)
        comp_reps = self.competition_agents(observe[...,:-1], observe, dgl_edges_comp)

        ### cooperation modeling ###
        adj_coop_dist = self.adj_coop_list.repeat(B,1,1)
        adj_coop_req = torch.cat([self.grid2adj_coop[grid_idx] for grid_idx in query_grids],dim=0)
        dgl_edges_coop = self.adj_to_dgledge(adj_coop_dist, adj_coop_req, self.distance, self.device)
        x_m = torch.cat([observe[...,:-1], comp_reps],dim=-1)
        coop_reps = self.cooperation_agents(x_m, x_m, dgl_edges_coop)
        observation = torch.cat([time_emb, observe[...,:-1], comp_reps, coop_reps],dim=-1)
        common_rep = self.actor_net(observation) # (B,N,hiddim)
        
        ### meta feature encoder ###
        obs_actionpre = torch.cat([observation, action_pre],dim=-1)
        h_t = self.gru_encoder(obs_actionpre.view(B*N,-1), h_pre.view(B*N,-1)).view(B,N,-1) # (B,N,hiddim)
        actions_tmp = self.meta_generator(h_t, common_rep)
        actions = torch.sigmoid(actions_tmp)
        # if(np.random.random() > 0.999):
        #     print('actor:{:.4f},{:.4f},{:.4f},{:.4f}'.\
        #           format(actions.max().item(),actions.mean().item(),actions.min().item(),actions.std(dim=1).mean().item()))
        return actions, h_t
    
class OUNoise(object):
    def __init__(self, args, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0, decay_period=30):
        self.mu = mu
        self.theta = theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.N = args.N
        
    def reset(self):
        self.state = np.ones(self.N,) * self.mu
        
    def evolve_state(self):
        x  = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.N)
        self.state = x + dx
        return self.state
    
    def action_noise(self, action, n_iter):
        n_q, K, _ = action.shape
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, n_iter / self.decay_period)
        ou = torch.from_numpy(ou_state).type(TorchFloat)
        ou = ou.view(1,K,1).repeat(n_q,1,1)/5
        ou = torch.clamp(ou, -0.1, 0.1)
        action = action + ou
        action = torch.clamp(action, 0, 1)
        return action
        