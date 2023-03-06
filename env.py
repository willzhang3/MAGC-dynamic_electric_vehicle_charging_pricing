import numpy as np
import torch

TorchFloat = None
TorchLong = None

class Charging_Env:
    def __init__(self, args, n_grids, supply_dist=None, demand_dist=None, cs_surgrids=None, \
            durations=None, distances=None, fees_24hour=None, powers=None, dpcs_mark=None, dpcs_id_set=None, operators=None):
        global TorchFloat, TorchLong
        TorchFloat = torch.cuda.FloatTensor if args.device == torch.device('cuda') else torch.FloatTensor
        TorchLong = torch.cuda.LongTensor if args.device == torch.device('cuda') else torch.LongTensor

        # env params
        self.grid2cs_duration = durations # (n_grids,N) -- the eta from each grid to cs
        self.grid2cs_distance = distances # (n_grids,N) -- the distance from each grid to cs
        self.cs_surgrids = cs_surgrids
        self.charge_fee24 = fees_24hour[...,0]
        self.electric_fee24 = fees_24hour[...,1]
        self.powers = powers
        self.N, self.n_grids = args.N, n_grids
        self.interval = args.interval 
        self.avg_charge_qt = args.avg_charge_qt
        self.std_charge_qt = args.std_charge_qt
        self.T = args.T_LEN*args.interval
        self.T_LEN = args.T_LEN
        self.n_pred = args.n_pred
        self.limit_waiting_time = self.n_pred*self.interval
        self.miss_time = args.miss_time
        self.gamma = args.gamma
        self.topk = args.topk
        self.power_rate = 0.5
        self.query_idx = 0
        self.sup_i = 1
        self.dpcs_mark = torch.from_numpy(dpcs_mark).type(TorchFloat) # (1,N,1) 1 if dynamic price, otherwise 0
        self.dpcs_id_set = dpcs_id_set

        # state features
        self.cs_idx = torch.from_numpy(np.arange(args.N)).type(TorchLong)
        self.operator = operators.reshape(1,self.N,1).repeat(self.T_LEN*self.interval,axis=0)
        self.powers_rec = powers.reshape(1,self.N,1).repeat(self.T_LEN*self.interval,axis=0)
        self.supply = np.zeros((self.T,self.N,1),dtype=np.int16) # supply at t
        self.supply_dist = supply_dist # (T*n_day,N,2)
        self.demand = np.zeros((self.T_LEN,n_grids,1),dtype=np.int16) # demand about future 15min
        self.demand_dist = demand_dist # (T_LEN,n_grids,1) 
        self.cs_demand = np.zeros((self.T_LEN*self.interval,self.N,1),dtype=np.int16) # demand in neighbor grids of cs
        self.t_step = np.arange(self.T_LEN,dtype=np.int16).reshape(self.T_LEN,1,1).repeat(self.N,axis=1) 
        self.t_step = self.t_step.repeat(self.interval,axis=0) # (T,N,1)
        self.chargefee = np.expand_dims(self.charge_fee24.transpose(1,0),axis=-1).repeat(4,axis=0)[:self.T_LEN].repeat(self.interval,axis=0)
        self.electricfee = np.expand_dims(self.electric_fee24.transpose(1,0),axis=-1).repeat(4,axis=0)[:self.T_LEN].repeat(self.interval,axis=0)
        self.querys = None
        self.base_state = None
        self._state = None
        
        # reset env
        self.RANDOM_SEED = 0

    def grid2allcs_durations(self, grid_idx, expand_dim=True):
        cs_duration = self.grid2cs_duration[grid_idx] # (N,)
        cs_duration = np.expand_dims(cs_duration,axis=-1) if expand_dim else cs_duration
        return cs_duration
    
    def grid2allcs_distances(self, grid_idx, expand_dim=True):
        cs_distance = self.grid2cs_distance[grid_idx] # (N,)
        cs_distance = np.expand_dims(cs_distance,axis=-1) if expand_dim else cs_distance
        return cs_distance
    
    def action_sample(self, ind):
        k = len(ind)
        idx = np.random.randint(k)
        return ind[idx]
    
    def est_power(self, cs_idx):
        charge_qt = np.random.normal(self.avg_charge_qt,self.std_charge_qt)
        return charge_qt

    def est_time(self, cs_idx):
        charge_qt = self.est_power(cs_idx)
        charge_time = (charge_qt / (self.powers[cs_idx]*self.power_rate)) * 60
        return charge_time, charge_qt

    def get_chargefee(self, cs_idx, hour):
        return self.charge_fee24[cs_idx,hour] # (N,24)
    
    def get_electricfee(self, cs_idx, hour):
        return self.electric_fee24[cs_idx,hour] # (N,24)
    
    def supply_load(self, day):
        """ Load real-world data
        """
        self.supply = self.supply_dist[day*self.T:day*self.T+self.T,:,:1]
        self.supply = np.where(self.supply>=0, self.supply, 0)
        return self.supply
    
    def demand_load(self, day):
        """ Load real-world querys
        """
        dates = ['20190518', '20190519', '20190520', '20190521', '20190522', '20190523', '20190524', '20190525', '20190526', '20190527', '20190528', '20190529', '20190530', '20190531',\
         '20190601', '20190602', '20190603', '20190604', '20190605', '20190606', '20190607', '20190608', '20190609', '20190610', '20190611', '20190612', '20190613', '20190614',\
         '20190615', '20190616', '20190617', '20190618', '20190619', '20190620', '20190621', '20190622', '20190623', '20190624', '20190625', '20190626', '20190627', '20190628',\
         '20190629', '20190630', '20190701']
        self.demand = self.demand.repeat(self.interval,axis=0) # demand about future 15min
        self.querys = [[] for t in range(self.T_LEN*self.interval)] # (T,[(grid_idx, query_time, query_idx)...])
        self.query_idx = 0
        with open("../exp_data_pricing/request/{}".format(dates[day]),"r") as fp:
            for line in fp:
                col = list(map(int,line.strip().split("\t")))
                grid_idx, t_min, target_cs= col[0], col[1], col[-1]
                query_tp = (grid_idx,t_min,self.query_idx,target_cs)
                if(t_min >= self.T_LEN*self.interval):
                    continue
                self.querys[col[1]].append(query_tp)
                self.query_idx += 1
                for delta in range(15):
                    tt = t_min - delta - 1
                    if(tt >= 0): self.demand[tt,grid_idx] += 1 
        return self.demand, self.querys
    
    def cs_neighbor_demand(self):
        for cs_idx in range(self.N):
            surgrid = self.cs_surgrids[cs_idx]
            for grid_idx in surgrid:
                self.cs_demand[:,cs_idx] += self.demand[:,grid_idx]
        return self.cs_demand
        
    def get_state(self, t):
        return self._state[t]

    def select_topk_cs(self, state, duration=None, inds=None, top_k=None):
        """ select top-k nearest cs
            Args:
                state (n_q,N,F)
                duration (n_q,N,1)
                inds (n_q,top_k,1)
                top_k (1,)
            Return:
                observe (n_q,top_k,F)
                inds (n_q,top_k,1)
        """
        if(top_k is None): top_k = self.topk
        if(inds is None):
            inds = np.argpartition(duration.squeeze(axis=-1), top_k-1)[...,:top_k] # (n_q, k)
            inds = torch.from_numpy(inds).type(TorchLong).unsqueeze(dim=-1)
        observe = state.gather(-2, inds.repeat(1,1,state.shape[-1]))
        return observe, inds

    def get_query(self, t):
        if(t<self.T):
            return self.querys[t]
        elif(t==self.T+self.limit_waiting_time-1): # a virtual query
            return [(0,self.T+self.limit_waiting_time-1,-1)] 

    def get_integrated_obs(self, st, et=None):
        if(et == None):
            return self._integrated_obs[st]
        return self._integrated_obs[st:et] # [m, int_obs(1,F)]
    
    def reset_randomseed(self, random_seed):
        self.RANDOM_SEED = random_seed
    
    def reset_state(self, state):
        self._state = state
        self._trajectory = [None for _ in range(self.query_idx)] # historical_rep and pre_action
        self._integrated_state = [None for _ in range(self.query_idx)] # each query corresponds to a state
        self._joint_action = [None for _ in range(self.query_idx)] # each query corresponds to a joint_action
        self._query_info = [None for _ in range(self.query_idx)] # info about each query
        self._query_grid = [None for _ in range(self.query_idx)] # query grid
        self.n_travel = np.zeros((self.N,1),dtype=np.int16) # # of traveling EVs
        print("n_query:",self.query_idx)

    def reset_event(self):
        self._event_arrival = [[] for _ in range(self.T+self.limit_waiting_time+1)] # (T,)
    
    def init_state(self):
        # state features
        self.supply = np.zeros((self.T,self.N,1),dtype=np.int16) # supply at t
        self.demand = np.zeros((self.T_LEN,self.n_grids,1),dtype=np.int16) # demand about future 15min
        self.cs_demand = np.zeros((self.T_LEN*self.interval,self.N,1),dtype=np.int16) # demand in neighbor grids of cs
        self.querys = None
        self.base_state = None

    def reset(self, RANDOM_SEED, day):
        self.reset_randomseed(RANDOM_SEED)
        np.random.seed(self.RANDOM_SEED)
        self.init_state()
        self.supply = self.supply_load(day)
        self.demand, self.querys = self.demand_load(day)
        self.cs_demand = np.clip(self.cs_neighbor_demand(),0,20) # (T,N,1)
        self.base_state = np.concatenate([self.powers_rec,self.supply,self.cs_demand,self.t_step,\
                              self.chargefee,self.electricfee,self.operator],axis=-1) #(T,N,F)
        # reset state and event_in 
        self.reset_state(self.base_state)
        self.reset_event()

    def arrival_step(self, t):
        """ process vehicle arrival at t
        """
        sup_i = self.sup_i # supply index in state
        t_event_arrival = self._event_arrival[t]  # list[(st, query_grid, query_idx, ...), ...]
        service_cnt, success_service_cnt = 0, 0 
        time_costs = []
        fee_costs = []
        revenues = []
        profits = []
        rewards = []
        for visit_tp in t_event_arrival:  # arrive at t
            st, query_grid, query_idx, action_duration, cs_idx, chargefee_dy, joint_action, o_cs_idx= visit_tp
            self.n_travel[cs_idx,0] -= 1
            ### error ###
            if(self.n_travel[cs_idx,0] < 0): 
                print("Error: n_travel is negative!")
            cs_idx = cs_idx.item()
            time_cost = max(t - st, action_duration)
            time_cost = time_cost if(time_cost<=self.limit_waiting_time) else self.miss_time 
            if(t<self.T):  # tn is the time vehicle in (or until exceed limit_waiting_time)
                done = 0; tn = t
            else: 
                done = 1; tn = t%self.T
            success_charge = False
            charge_time = 0
            t_supply = self._state[tn,cs_idx,sup_i]
            #### simulate when vehicle in or fail ###
            while(time_cost<=self.limit_waiting_time):
                if(self._state[tn,cs_idx,sup_i]>=1): # vehicle success charge
                    in_t = tn
                    charge_time, charge_qt = np.ceil(self.est_time(cs_idx)).astype(np.int32) 
                    leave_t = tn + charge_time # leave time 
                    self._state[in_t:leave_t,cs_idx,sup_i] -= 1
                    if(leave_t > self.T): 
                        self._state[:leave_t%self.T,cs_idx,sup_i] -= 1
                    success_charge = True
                    break
                else:  # queue and wait this minute
                    self._state[tn,cs_idx,sup_i] -= 1
                    tn += 1
                    if(tn>=self.T):
                        tn = tn % self.T
                        done = 1
                    time_cost += 1 
            
            """ reward design """
            reward_profit = 0
            if(success_charge): 
                if(done == 1): tn += self.T
                charge_fee = chargefee_dy[0,cs_idx,0].item()
                electric_fee = self.get_electricfee(cs_idx, hour=int(tn%self.T/60))
                if(cs_idx in self.dpcs_id_set):
                    ### metric ###
                    fee_costs.append(charge_fee)
                    revenue = charge_fee*charge_qt
                    revenues.append(revenue)
                    profit = (charge_fee-electric_fee)*charge_qt
                    profits.append(profit)
                    success_service_cnt += 1 # successful service count
                    reward_profit = charge_fee - electric_fee
                reward = reward_profit
            else: 
                leave_t = st + time_cost
                time_cost = self.miss_time
                tn = st+self.limit_waiting_time+1
                reward = 0
                    
            if(cs_idx in self.dpcs_id_set):
                reward += 0.1
                time_costs.append(time_cost)
                service_cnt += 1

            self._query_info[query_idx] = [cs_idx, reward, done]
            rewards.append(reward)

        self._event_arrival[t] = [] # clear arrival event at t
        return fee_costs, revenues, profits, time_costs, service_cnt, success_service_cnt, rewards
            
    def query_step(self, n_q, t_querys, action_cs_idx, chargefee_dy, joint_action, original_cs_idx):
        """ process charging query at t
        """
        for i in range(n_q):
            cs_idx = action_cs_idx[i]
            if(cs_idx == -1): continue
            query_tp = t_querys[i]
            st, query_grid, query_idx = query_tp[1], query_tp[0], query_tp[2]
            self._query_grid[query_idx] = query_grid
            o_cs_idx = original_cs_idx[i]
            cs_durations = self.grid2allcs_durations(query_grid, expand_dim=False) # (N,)  [1,60]
            action_duration = cs_durations[cs_idx] 
            cs_eta = st + action_duration
            if(cs_eta > self.T+self.limit_waiting_time -1): cs_eta = self.T+self.limit_waiting_time -1 # end of day
            visit_tp = (st, query_grid, query_idx, action_duration, cs_idx, chargefee_dy[i:i+1], joint_action[i:i+1], o_cs_idx)
            self._event_arrival[cs_eta].append(visit_tp)
            
    def transition_step(self, q_idx):
        """ derive transition
        """
        if(q_idx >= self.query_idx):
            return 0, None

        ### transition ###
        next_q_idx = (q_idx+1)%self.query_idx
        query_info = self._query_info[q_idx] # [cs_idx, reward, done]
        next_state = self._integrated_state[next_q_idx]
        if((query_info is None) or (next_state is None)):
            return 0, None
        states = []
        previous_reps = []
        durations = []
        joint_actions = []
        rewards = []
        next_states = []
        next_previous_reps = []
        next_durations = []
        dones = []
        etc = []
        cs_idx, reward, done = query_info
        rewards.append(reward)
        dones.append(done)
        etc.append([cs_idx,self._query_grid[q_idx],self._query_grid[next_q_idx]])
        state = self._integrated_state[q_idx]
        previous_rep = torch.cat(self._trajectory[q_idx-1],dim=-1)
        previous_reps.append(previous_rep)
        states.append(state)
        cs_duration = self.grid2allcs_durations(self._query_grid[q_idx])
        durations.append(cs_duration)
        joint_action = self._joint_action[q_idx]
        joint_actions.append(joint_action)
        next_states.append(next_state)
        next_previous_rep = torch.cat(self._trajectory[q_idx],dim=-1)
        next_previous_reps.append(next_previous_rep)
        next_cs_duration = self.grid2allcs_durations(self._query_grid[next_q_idx])
        next_durations.append(next_cs_duration)

        return len(states), (states, previous_reps, joint_actions, rewards, next_states, next_previous_reps, dones, etc, durations, next_durations)
    
    