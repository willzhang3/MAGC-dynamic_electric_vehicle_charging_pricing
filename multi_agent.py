import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import math
from torch.distributions import Categorical, Normal
from collections import deque
from net import ReplayBuffer,Graphpool,Critic,Actor,OUNoise
import sys
sys.path.append('./user_model/')
from user_net import UserModel

TorchFloat = None
TorchLong = None

class Agent_MAGC(nn.Module):
    def __init__(self, env, args, cs_tuple, adjs, writer, fw_summary, LOAD_PATH=None):
        """ MAGC
        """
        super(Agent_MAGC, self).__init__()
        global TorchFloat,TorchLong
        TorchFloat = torch.cuda.FloatTensor if args.device == torch.device('cuda') else torch.FloatTensor
        TorchLong = torch.cuda.LongTensor if args.device == torch.device('cuda') else torch.LongTensor
        
        self.fee_i = 4
        self.env = env
        self.writer = writer
        self.fw_summary = fw_summary
        self.device = args.device
        self.gamma, self.lr_a, self.lr_c = args.gamma, args.lr_a, args.lr_c
        self.soft_tau_a, self.soft_tau_c = args.soft_tau_a, args.soft_tau_c
        self.T_LEN, self.N = args.T_LEN, args.N
        self.n_pred, self.miss_time, self.interval = args.n_pred, args.miss_time, args.interval
        self.T = self.T_LEN*self.interval
        self.limit_waiting_time = args.n_pred*args.interval
        self.clip_norm = args.clip_norm
        self.batch_size = args.batch_size
        self.noise = args.noise
        self.supply_clip = 10
        self.pre_reward = 0
        self.hiddim = args.hiddim
        dpcs_mark, dpcs_id, spcs_id = cs_tuple
        self.dpcs_mark = torch.from_numpy(dpcs_mark).type(TorchFloat) # (1,N,1) 1 if dynamic price, otherwise 0
        self.dpcs_mark_batch = self.dpcs_mark.repeat(args.batch_size,1,1)
        self.dpcs_id = dpcs_id
        self.spcs_id = spcs_id
        self.feescale = args.feescale
        self.topk = args.topk
        self.beta = args.beta
        self.momentum = args.momentum
        self.noise_std = args.std
        self.debug = args.debug
        
        # buffer & noise
        self.replay_buffer = ReplayBuffer(args.buffer_size)
        self.ouNoise = OUNoise(args)
    
        # networks: Critic (Q_funciton), Actor
        self.Critic = Critic(env, args).to(self.device)
        if(args.load == True):
            self.Critic.load_state_dict(torch.load(LOAD_PATH)['critic'])
        self.Critic_target = Critic(env, args).to(self.device)
        self.Critic_target.load_state_dict(self.Critic.state_dict())
        self.Critic_target.eval()
        self.Actor = Actor(env, args, adjs).to(self.device)
        if(args.load == True):
            self.Actor.load_state_dict(torch.load(LOAD_PATH)['actor'])
            print("Loading parameters: {}".format(LOAD_PATH))
        self.Actor_target = Actor(env, args, adjs).to(self.device)
        self.Actor_target.load_state_dict(self.Actor.state_dict())
        self.Actor_target.eval()
        self.Graphpool = Graphpool(env, args, adjs, cs_tuple).to(self.device)
        if(args.load == True):
            self.Graphpool.load_state_dict(torch.load(LOAD_PATH)['graphpool'])
        
        # UserModel
        self.UserNet = UserModel(args).to(self.device)
        UserModel_PATH = "./user_model/params/user_model.pkl"
        self.UserNet.load_state_dict(torch.load(UserModel_PATH)['actor'])
        self.UserNet.eval()
        
        self.previous_rep = [torch.zeros((1,args.N,args.hiddim)).type(TorchFloat), torch.zeros((1,args.N,1)).type(TorchFloat)]
        self.previous_rep_train = [torch.zeros((1,args.N,args.hiddim)).type(TorchFloat), torch.zeros((1,args.N,1)).type(TorchFloat)]
        
        # loss and optimizer
        self.mseloss = nn.MSELoss()
        self.logceloss = nn.CrossEntropyLoss()
        if(args.opt=="sgd"):
            self.optimizer_critic = torch.optim.SGD([{'params':self.Critic.parameters(),'lr':args.lr_c},\
                                        {'params':self.Graphpool.parameters(),'lr':args.lr_g}],\
                                        lr=args.lr_c, momentum=self.momentum, weight_decay=args.wdecay)
            self.optimizer_actor = torch.optim.SGD(self.Actor.parameters(),\
                    lr=args.lr_a, momentum=self.momentum, weight_decay=args.wdecay)
        
        else:
            self.optimizer_critic = torch.optim.Adam(self.Critic.parameters(),\
                    lr=args.lr_c, betas=(0.9, 0.99), eps=args.eps)
            self.optimizer_actor = torch.optim.Adam(self.Actor.parameters(),\
                    lr=args.lr_a, betas=(0.9, 0.99), eps=args.eps)
    
    def reset_agent(self):
        self.LASTUPDATE_step = 0
        self.env._trajectory[-1] = self.previous_rep

    def stack_transition(self, transition, is_tensor=True):
        np_trans = []
        for ele in transition:
            if(is_tensor):
                np_trans.append(torch.cat(ele,dim=0))
            else:
                np_trans.append(np.stack(ele))
        return np_trans

    def state_normalization(self, state):
        # powers,supply,cs_demand,t_step,chargefee,electricfee,operator,duration,distance,n_travel
        power = state[...,:1].div(120)
        supply = state[...,1:2]
        supply_cp = torch.where(supply<=self.supply_clip, supply, self.supply_clip*torch.ones(1,1).to(self.device)).div(self.supply_clip)
        demand = state[...,2:3].div(20)
        t_step = state[...,3:4]
        chargefee = state[...,4:5]
        electricfee = state[...,5:6]
        operator = state[...,6:7]
        duration = state[...,7:8].div(self.miss_time)
        distance = state[...,8:9].div(20)
        travel_count = state[...,9:10].div(self.supply_clip)
        supply_rest = (supply_cp - travel_count)

        # agent joint observations
        norm_state_actor = torch.cat([t_step,operator,supply_cp,demand,power,travel_count,\
                              duration,distance,supply_rest,electricfee,chargefee],dim=-1) # (n_q, N, F1)
    
        # user state
        norm_state_user = torch.cat([t_step,travel_count,distance,supply_cp,demand,power,chargefee,duration],dim=-1) # (n_q, N, F2)
        return norm_state_actor, norm_state_user
    
    def action_estimation(self, t_querys, t, norm_state, chargefee, electricfee, duration, n_iter, is_val=False):
        """ action estimation (in batch)
        """
        norm_state_actor, norm_state_user = norm_state
        n_q = len(t_querys)
        query_idxs = [t_querys[i][2] for i in range(n_q)]
        query_grids = [t_querys[i][0] for i in range(n_q)]
        previous_reps = [self.previous_rep[0].repeat(n_q,1,1), self.previous_rep[1].repeat(n_q,1,1)]
        global_actions, h_t = self.Actor(norm_state_actor, query_grids, previous_reps) 
        global_actions = global_actions.detach() # (n_q, N, 1) dynamic charging price
        h_t = h_t.detach()
        if((not is_val) and self.noise):
            # global_actions = self.ouNoise.action_noise(global_actions, n_iter) # ou noise
            nz = torch.normal(torch.zeros_like(global_actions),torch.ones_like(global_actions)*self.noise_std).clip(-0.1,0.1)
            global_actions = torch.clamp(global_actions+nz,0,1)
        scale_actions = global_actions*self.feescale
        chargefee_dy = chargefee.clone()
        chargefee_dy[:,self.dpcs_id,:] = scale_actions[:,self.dpcs_id,:]
        norm_state_user[...,-2:-1] = chargefee_dy
        user_actions = self.UserNet(norm_state_user).detach() # user response
        action_cs_idx = []
        original_cs_idx = []
        for i, query_tp in enumerate(t_querys):
            o_cs_idx = query_tp[-1]
            original_cs_idx.append(o_cs_idx)
            # user response #
            select_cs_idx = torch.argmax(user_actions[i].squeeze(dim=-1), dim=-1).item() # int
            self.env.n_travel[select_cs_idx,0] += 1
            action_cs_idx.append(select_cs_idx)
        action_cs_idx = torch.from_numpy(np.asarray(action_cs_idx)).type(TorchLong)
        mean_ht = h_t.mean(axis=0).unsqueeze(dim=0) # mean h_t of n_q queries
        mean_actions = global_actions.mean(axis=0).unsqueeze(dim=0) # mean actions of n_q queries
        self.previous_rep = [mean_ht, mean_actions] 
        for i, q_idx in enumerate(query_idxs):
            self.env._trajectory[q_idx] = [mean_ht, mean_actions] # historical trajectory
            self.env._integrated_state[q_idx] = norm_state_actor[i].unsqueeze(dim=0)
            self.env._joint_action[q_idx] = global_actions[i].unsqueeze(dim=0)
        for i, q_idx in enumerate(query_idxs):
            if(i==0): continue
            cs_idx = action_cs_idx[i-1]
            self.env._integrated_state[q_idx] = self.env._integrated_state[q_idx-1].clone()
            self.env._integrated_state[q_idx][0,cs_idx,5] = self.env._integrated_state[q_idx-1][0,cs_idx,5] + 1/self.supply_clip
            self.env._integrated_state[q_idx][0,cs_idx,8] = self.env._integrated_state[q_idx-1][0,cs_idx,8] - 1/self.supply_clip
            
        return action_cs_idx, original_cs_idx, chargefee_dy, global_actions
    
    def step(self, cur_t, n_iter, is_val=False): # one time_step
        if(cur_t==0 and self.noise): self.ouNoise.reset()
        fee_i = self.fee_i
        losses_critic,losses_actor,rec_reward = [],[],[]
        fee_costs,revenues,profits,time_costs= [],[],[],[]
        n_query, n_service, n_success_service = 0,0,0 
        st_minute = cur_t*self.interval
        ed_minute = st_minute+self.interval
        if(cur_t == self.T_LEN-1): ed_minute += self.limit_waiting_time
        for t in range(st_minute, ed_minute):
            if(self.debug and t > 60):
                break
            """ event1: handle vehicle arrival at t
            """
            t_feecost, t_revenue, t_profit, t_time_cost, service_cnt, success_service_cnt, t_reward = self.env.arrival_step(t) # (n_q,)
            fee_costs.extend(t_feecost)
            revenues.extend(t_revenue)
            profits.extend(t_profit)
            time_costs.extend(t_time_cost)
            rec_reward.extend(t_reward)
            n_service += service_cnt
            n_success_service += success_service_cnt
            if(t>=self.T and t!=self.T+self.limit_waiting_time-1): continue
            n_step = n_iter*self.T+t
            if(len(t_reward)>0):
                self.pre_reward = np.mean(t_reward)
            """ event2: handle charging query at t
            """
            t_querys = self.env.get_query(t) # (n_q,4) # a tuple list, [tuple(grid_idx, query_time, query_idx, target_cs),...,]
            n_q = len(t_querys)
            n_query += n_q
            if(n_q > 0 and t<self.T):
                t_state = torch.from_numpy(self.env.get_state(t)).type(TorchFloat).repeat(n_q,1,1) # (n_q,N,F)
                # one query to all cs
                chargefee = t_state[...,fee_i:fee_i+1] # (n_q,N,1)
                electricfee = t_state[...,fee_i+1:fee_i+2] # (n_q,N,1)
                duration = np.asarray([self.env.grid2allcs_durations(query_tp[0]) for query_tp in t_querys]) # (n_q,N,1) 
                nq_duration = torch.from_numpy(duration).type(TorchFloat)
                distance = np.asarray([self.env.grid2allcs_distances(query_tp[0]) for query_tp in t_querys]) # (n_q,N,1) 
                nq_distance = torch.from_numpy(distance).type(TorchFloat)
                nq_n_travel = torch.from_numpy(self.env.n_travel).type(TorchFloat).unsqueeze(dim=0).repeat(n_q,1,1)
                primal_state = torch.cat([t_state,nq_duration,nq_distance,nq_n_travel],dim=-1) #(n_q,N,F)
                norm_state = self.state_normalization(primal_state) #(n_q,N,F) 
                ### take action acoording to observations
                action_cs_idx, original_cs_idx, chargefee_dy, global_actions = self.action_estimation(t_querys, t, norm_state, chargefee, electricfee, duration, n_iter, is_val) # (n_q,)
                self.env.query_step(n_q, t_querys, action_cs_idx, chargefee_dy, global_actions, original_cs_idx)

            """ *** derive transition *** 
            """
            if(not is_val):
                while(True):
                    # transition: (states, previous_reps, joint_actions, rewards, next_states, next_previous_reps, dones, etc, durations, next_durations)
                    n_trans, transitions = self.env.transition_step(self.LASTUPDATE_step)
                    if(n_trans>0):
                        ### add to replay buffer
                        self.replay_buffer.push(transitions)
                        self.LASTUPDATE_step += 1
                    else:
                        break
                    
            """ *** model update *** 
            """
            if(len(self.replay_buffer) >= self.batch_size and not is_val):
                states, previous_reps, joint_actions, rewards, next_states, next_previous_reps, dones, etc, durations, next_durations = \
                        self.replay_buffer.sample(self.batch_size) # (batch_size,)
                ###########  =======================================  ##############   
                states, previous_reps, joint_actions, next_states, next_previous_reps \
                        = self.stack_transition((states, previous_reps, joint_actions, next_states, next_previous_reps),is_tensor=True)
                rewards, dones, etc, durations, next_durations \
                        = self.stack_transition((rewards, dones, etc, durations, next_durations),is_tensor=False)
                rewards = torch.from_numpy(rewards).type(TorchFloat).view(-1,1) # (B,1)
                rewards = torch.clamp(rewards,0,10)
                dones = torch.from_numpy(dones).type(TorchFloat).view(-1,1)
                etc = torch.from_numpy(etc).unsqueeze(dim=-1)
                cs_idxs = etc[:,:1,:].type(TorchLong)
                grid_idxs = etc[:,1].squeeze().type(TorchLong)
                next_grid_idxs = etc[:,2].squeeze().type(TorchLong)
                
                """critic update"""
                ### rl loss ###
                # state and action
                state_actions = states.clone()
                state_actions[:,self.dpcs_id,-1] = joint_actions[:,self.dpcs_id,0]*self.feescale
                market_rep = self.Graphpool(state_actions, grid_idxs, self.dpcs_mark_batch) # (B,1)
                q_values = self.Critic(market_rep)
                # next state and action
                next_state_actions = next_states.clone()
                next_previous_reps = [next_previous_reps[...,:self.hiddim], next_previous_reps[...,self.hiddim:]]
                next_actions, _ = self.Actor_target(next_states, next_grid_idxs, next_previous_reps)
                next_actions = next_actions.detach()
                next_state_actions[:,self.dpcs_id,-1] = next_actions[:,self.dpcs_id,0]*self.feescale
                next_market_rep = self.Graphpool(next_state_actions, next_grid_idxs, self.dpcs_mark_batch).detach()
                next_q_values = self.Critic_target(next_market_rep).detach()
                next_q_values = torch.clamp(next_q_values,0,100)
                expected_returns = rewards + self.gamma*next_q_values*(1-dones) # (B,1)
                rl_loss = self.mseloss(q_values, expected_returns)

                ### graph contrastive loss ###
                # select the nearest top-k agents to the charging request
                state_action_topk, inds = self.env.select_topk_cs(state_actions, durations, inds=None, top_k=self.topk)
                dpcs_mark_topk, _ = self.env.select_topk_cs(self.dpcs_mark_batch, durations, inds=None, top_k=self.topk)
                # query instance representation
                market_rep = self.Graphpool(state_action_topk, grid_idxs, dpcs_mark_topk, inds) # (B,1)
                # positive and negative instance representations
                shuffle_idx = np.arange(self.batch_size)
                np.random.shuffle(shuffle_idx)
                durations_shuffle = durations[shuffle_idx]
                state_action_topk_keys, inds_keys = self.env.select_topk_cs(state_actions, durations_shuffle, inds=None, top_k=self.topk)
                dpcs_mark_topk_keys, _ = self.env.select_topk_cs(self.dpcs_mark_batch, durations_shuffle, inds=None, top_k=self.topk)
                market_rep_keys = self.Graphpool(state_action_topk_keys, grid_idxs, dpcs_mark_topk_keys, inds_keys).detach() # (B,1)

                # if(np.random.random() > 0.998):
                #     print("{:.4f},{:.4f},{:.4f},{:.4f}".format(q_values.mean().item(),expected_returns.mean().item(),\
                #       rewards.mean().item(),next_q_values.mean().item()))

                market_rep_keys = self.Graphpool.c_weight(market_rep_keys) # (B, d)
                logits = torch.mm(market_rep, market_rep_keys.transpose(1,0)) # (B,B)
                logits = logits - logits.max(dim=-1,keepdim=True).values # for softmax stability
                labels = torch.arange(0,logits.shape[0]).type(TorchLong)
                gc_loss = self.logceloss(logits, labels)

                ### overall critic loss ###
                critic_loss = rl_loss + self.beta * gc_loss

                # if(np.random.random() > 0.998):
                #     print(self.mseloss(q_values, expected_returns),self.logceloss(logits, labels))
                #     print(torch.softmax(logits,dim=-1).max(dim=-1))

                # Critic and graph pooling parameters update
                critic_loss_item = self.update_critic(critic_loss)
                self.writer.add_scalar('critic_loss', critic_loss_item, n_step)
                losses_critic.append(critic_loss_item)

                """actor update"""
                previous_reps = [previous_reps[...,:self.hiddim], previous_reps[...,self.hiddim:]]
                new_actions, _ = self.Actor(states, grid_idxs, previous_reps) # (B,N,1)
                new_state_actions = states.clone()
                new_state_actions[:,self.dpcs_id,-1] = new_actions[:,self.dpcs_id,0]*self.feescale
                market_rep_new = self.Graphpool(new_state_actions, grid_idxs, self.dpcs_mark_batch)
                new_q_values = self.Critic(market_rep_new)
                actor_loss = - torch.mean(new_q_values)
                critic_values = {"q_value":q_values.mean().item(),
                           "new_q_value":new_q_values.mean().item(),
                            }
                self.writer.add_scalars('critic_values', critic_values, n_step)
                
                # Actor parameters update
                actor_loss_item = self.update_actor(actor_loss) 
                losses_actor.append(actor_loss_item)
                self.writer.add_scalar('actor_loss', actor_loss_item, n_step)
                
                self.soft_update(self.Critic_target, self.Critic, self.soft_tau_c)
                self.soft_update(self.Actor_target, self.Actor, self.soft_tau_a)
                self.writer.add_scalar('reward', self.pre_reward, n_step)
                self.writer.flush()

        return fee_costs, profits, revenues, time_costs, losses_critic, losses_actor, n_query, n_service, n_success_service, rec_reward
    
    def update_critic(self, loss):
        """ Update critic params by gradient descent. """
        self.optimizer_critic.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.Graphpool.parameters(), self.clip_norm)
        nn.utils.clip_grad_norm_(self.Critic.parameters(), self.clip_norm)
        self.optimizer_critic.step()
        return loss.item()

    def update_actor(self, loss):
        """ Update actor params by gradient descent. """
        self.optimizer_actor.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.Actor.parameters(), self.clip_norm)
        self.optimizer_actor.step()
        return loss.item()

    def soft_update(self, target, src, soft_tau):
        for target_param, param in zip(target.parameters(), src.parameters()):
            target_param.data.copy_(target_param.data*(1.0-soft_tau) + param.data*soft_tau)

