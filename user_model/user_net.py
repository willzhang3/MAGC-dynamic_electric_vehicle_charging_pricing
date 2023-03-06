import torch
import torch.nn as nn
import numpy as np
import random
import math
from utils import *

class UserModel(nn.Module):
    def __init__(self, args):
        """Initialization."""
        super(UserModel, self).__init__()
        global TorchFloat,TorchLong
        TorchFloat = torch.cuda.FloatTensor if args.device == torch.device('cuda') else torch.FloatTensor
        TorchLong = torch.cuda.LongTensor if args.device == torch.device('cuda') else torch.LongTensor
        self.device = args.device
        self.time_dim = 4
        self.time_emb = nn.Embedding(args.T_LEN, self.time_dim, _weight=torch.rand(args.T_LEN,self.time_dim))
        self.actor_net = nn.Sequential(
            nn.Linear(self.time_dim+7, 64), 
            nn.ReLU(inplace=True),
            nn.Linear(64, 64), 
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            )
        
    def forward(self, observe):
        """Forward method implementation."""
        time_emb = self.time_emb(observe[...,0].type(TorchLong))
        observe = observe[...,1:]
        observe = torch.cat([time_emb, observe],dim=-1)
        actions = self.actor_net(observe) # (batch,N)
        return actions
