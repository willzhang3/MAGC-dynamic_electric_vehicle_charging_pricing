import math
import torch
import time
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import dgl

class SpGraphAttentionLayer(nn.Module):
    def __init__(self, args, in_feat, nhid, graphtype):
        super(SpGraphAttentionLayer, self).__init__()
        self.w_att= nn.Linear(2*in_feat+3-(graphtype=="competition"), nhid, bias=True)
        self.va = nn.Parameter(torch.zeros(1,nhid))
        nn.init.normal_(self.va.data)
        self.mlp = nn.Sequential(
            nn.Linear(in_feat+3, nhid, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(nhid, nhid, bias=True),
            nn.ReLU(inplace=True)
        )
    
    def edge_attention(self, edges):
        # edge UDF
        att_sim = torch.sum(self.va*torch.tanh(self.w_att(torch.cat([edges.src['h_value'], edges.dst['h_key'],\
                     edges.data['feature']],dim=-1))),dim=-1)
        return {'att_sim': att_sim}

    def message_func(self, edges):
        # message UDF
        return {'h_value': edges.src['h_value'], 'att_sim': edges.data['att_sim'], 'f_edge': edges.data['feature']}

    def reduce_func(self, nodes):
        # reduce UDF
        alpha = F.softmax(nodes.mailbox['att_sim'], dim=1) # (# of nodes, # of neibors)
        alpha = alpha.unsqueeze(-1)
        nodes_msgs = torch.cat([nodes.mailbox['h_value'], nodes.mailbox['f_edge']],dim=-1)
        h_att = torch.sum(alpha * nodes_msgs, dim=1)
        return {'h_att': h_att}

    def forward(self, X_key, X_value, g):
        """
        :param X_key: X_key data of shape (B, num_nodes(N), in_features_1).
        :param X_value: X_value dasta of shape (B, num_nodes(N), in_features_2).
        :param g: sparse graph.
        :return: Output data of shape (B, num_nodes(N), out_features).
        """
        B, N, in_feat = X_key.size()
        h_key = X_key.view(B*N,-1)
        h_value = X_value.view(B*N,-1)
        g.ndata['h_key'] = h_key
        g.ndata['h_value']= h_value
        g.apply_edges(self.edge_attention)
        g.update_all(self.message_func, self.reduce_func)
        h_conv = g.ndata.pop('h_att') # (N,out_features)
        h_conv = self.mlp(h_conv)
        return h_conv.view(B,N,-1)
        
class GAT(nn.Module):
    def __init__(self, args, in_feat, nhid, graphtype=None, dropout=0):
        """sparse GAT."""
        super(GAT, self).__init__()
        self.device = args.device
        self.dropout = nn.Dropout(dropout)
        self.att_layer = SpGraphAttentionLayer(args, in_feat, nhid, graphtype)

    def forward(self, X_key, X_value, adj): # no self-loop
        out = self.dropout(self.att_layer(X_key, X_value, adj))
        return out
