import torch
import argparse
import logging
import numpy as np
import json
import time
import os
import shutil
import pickle
import pandas as pd
from env import Charging_Env
from multi_agent import Agent_MAGC
from utils import *
from tensorboardX import SummaryWriter

################################### Initialize Hyper-Parameters ###################################
parser = argparse.ArgumentParser(description='MAGC')
parser.add_argument('--gpu', type=str, default="0", help='Which GPU to use.')
parser.add_argument('--state', type=str, default="def", help='The state of this running.')
parser.add_argument('--logmode', type=str, default="a", help='File mode of logging.')
parser.add_argument('--debug', action="store_true", default=False, help='Debug.')
parser.add_argument('--encuda', action="store_false", default=True, help='Enable CUDA training.')
parser.add_argument('--summary', action="store_false", default=True, help='SummaryWriter.')
parser.add_argument('--seed', type=int, default=3, help='Random seed.')
parser.add_argument('--noise', action="store_true", default=False, help='Add noise to action.')
parser.add_argument('--std', type=float, default=0.05, help='noise std.')
parser.add_argument('--n_pred', type=int, default=3, help='Max time_cost step the query not miss.')
parser.add_argument('--T_LEN', type=int, default=96, help='Number of time steps.')
parser.add_argument('--miss_time', type=int, default=46, help='Penalty time_cost if query miss.')
parser.add_argument('--interval', type=int, default=15, help='Time interval of one time step.')
parser.add_argument('--avg_charge_qt', type=float, default=48.96, help='Statistical Avg of charge.')
parser.add_argument('--std_charge_qt', type=float, default=10.43, help='Statistical Std of charge.')
parser.add_argument('--N', type=int, default=-1, help='Number of charging station.')
parser.add_argument('--hiddim', type=int, default=64, help='Dimension of NN.')
parser.add_argument('--eps', type=float, default=1e-5, help='eps.')
parser.add_argument('--dist_eps', type=float, default=2000, help='adjacent distance threshold.')
parser.add_argument('--dist_norm', type=float, default=5000, help='adjacent distance normalization.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout.')
parser.add_argument('--normalization', type=str, default="min-max", choices=['z-score', 'min-max'], help='File mode of logging.')
parser.add_argument('--com_dim', type=int, default=16, help='Output dim of communications.')
parser.add_argument('--feescale', type=float, default=2.3, help='charging fee scale')
parser.add_argument('--k_c', type=float, default=0.8, help='k_c rate.')
parser.add_argument('--k_h', type=float, default=0.15, help='k_h rate.')
parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor in TD.')
parser.add_argument('--lr_c', type=float, default=1e-2, help='Initial learning rate.')
parser.add_argument('--lr_a', type=float, default=1e-3, help='Initial learning rate.')
parser.add_argument('--lr_g', type=float, default=1e-1, help='Initial learning rate.')
parser.add_argument('--wdecay', type=float, default=0, help='weight_decay.')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum.')
parser.add_argument('--soft_tau_c', type=float, default=1e-3, help='Soft update ratio in critic params.')
parser.add_argument('--soft_tau_a', type=float, default=1e-3, help='Soft update ratio in actor params.')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
parser.add_argument('--buffer_size', type=int, default=2000, help='Buffer capacity.')
parser.add_argument('--opt', type=str, default="sgd", choices=['sgd','adam'], help='optimizor.')
parser.add_argument('--beta', type=float, default=0.5, help='loss coefficient of contrastive loss.')
parser.add_argument('--clip_norm', type=float, default=0.5, help='clip param.')
parser.add_argument('--temp', type=float, default=1, help='temperature.')
parser.add_argument('--load', action="store_true", default=False, help='Load parameters.')
parser.add_argument('--load_path', type=str, default="def", help='load path.')

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
file_name = 'MAGC_{}'.format(args.state)
if(args.debug == True):
    file_name = 'MAGC_debug'
print("logfile:",file_name)
logging.basicConfig(level = logging.INFO,filename='../logs/{}.log'.format(file_name),filemode='{}'.format(args.logmode),format = '%(message)s')

if(args.summary):
    if(not os.path.exists("./runs/")):
        os.mkdir("./runs/")
    tblog_dir = "./runs/{}".format(args.state)
else:
    if(not os.path.exists("./runs/")):
        os.mkdir("./runs/")
    tblog_dir = "./runs/debug"
if(os.path.exists(tblog_dir)):
    shutil.rmtree(tblog_dir)
if(os.path.exists(tblog_dir+".txt")):
    os.remove(tblog_dir+".txt")
fw_summary = open(tblog_dir+".txt","a")
writer = SummaryWriter(log_dir=tblog_dir)

logger = logging.getLogger(__name__)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.encuda and torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')
args.pid = os.getpid()

################################### Load datas ###################################
PATH_DATA = "../exp_data_pricing/"
# (N_DAY*T,N,2) -- supply at t
supply_dist = np.load(os.path.join(PATH_DATA,"20190518-20190701_supply.npy")).transpose(1,0,2)
# (n_grids,N) -- the eta (in second) from each grid to all cs
durations = np.load(os.path.join(PATH_DATA,"durations.npy"))
durations = np.clip(np.ceil(durations/60).astype(np.int32),0,args.miss_time)
# (n_grids,N) -- the distance (in meter) from each grid to all cs
distances = np.load(os.path.join(PATH_DATA,"distances.npy"))
distances = np.clip(distances/1000,0,20)
# (N, grid_ids)
with open(os.path.join(PATH_DATA,"cs_surgrids.list"),"r") as fp:
    cs_surgrids = np.asarray(json.load(fp))
# (N, 24)
fee_24hour = np.load(os.path.join(PATH_DATA,"fees_24hour.npy"))
electricity_24hour = np.load(os.path.join(PATH_DATA,"electricity_24hour.npy"))
fees_24hour = np.stack([fee_24hour, electricity_24hour],axis=-1)
# (N, 2)
with open(os.path.join(PATH_DATA,"power_tp.list"),"r") as fp:
    power_tp = json.load(fp)
power_tp = np.asarray(power_tp,dtype=np.int32)
powers, operators = power_tp[:,0], power_tp[:,1]
operator_map = {}
tp_idx = 0
for i,tp in enumerate(operators):
    if(tp not in operator_map):
        operator_map[tp] = tp_idx
        tp_idx += 1
    operators[i] = operator_map[tp]
args.n_operator = tp_idx
args.N = supply_dist.shape[1]
args.topk = int(args.N*args.k_c+0.5)
n_grids = durations.shape[0]
print(args)
logger.info(args)

# cs with dynamic price
with open(os.path.join(PATH_DATA,"tp_indexes/idx_telaidian.list"),"r") as fp:
    dpcs_id = json.load(fp)
print("# of dynamic price cs:",len(dpcs_id))
dpcs_id_set = set(dpcs_id)
spcs_id_set = set(range(args.N)) - dpcs_id_set
spcs_id = list(spcs_id_set)
dpcs_mark = np.zeros((1,args.N,1))
dpcs_mark[:,dpcs_id,:] = 1

# (N, N)
distance_matrix = np.load(os.path.join(PATH_DATA,"distance_matrix.npy")) 
# (n_grids, N, N)
grid2adj_comp = np.load(os.path.join(PATH_DATA,"grid2denseadj_comp.npy"))
grid2adj_coop = np.load(os.path.join(PATH_DATA,"grid2denseadj_coop.npy"))
adj_comp_list, adj_coop_list, grid2adj_comp, grid2adj_coop, distance_norm = adj_matrixs(args, distance_matrix, dpcs_id, spcs_id, grid2adj_comp, grid2adj_coop, operators)
adjs = (adj_comp_list, adj_coop_list, grid2adj_comp, grid2adj_coop, distance_norm)

LOAD_PATH = "params/{}.pkl".format(args.load_path)
################################### Initialize env and agent ###################################
env = Charging_Env(args, n_grids, supply_dist, None, cs_surgrids, durations, distances, fees_24hour, powers, dpcs_mark, dpcs_id_set, operators)
agent = Agent_MAGC(env, args, (dpcs_mark, dpcs_id, spcs_id), adjs, writer, fw_summary, LOAD_PATH)

################################### Training ###################################
MAX_ITER = 60
N_DAY_TRAIN = 28
day_shuffle = []
for i in range(np.ceil(MAX_ITER/N_DAY_TRAIN).astype(np.int32)):
    days = list(range(N_DAY_TRAIN))
    np.random.shuffle(days)
    day_shuffle += days
    
max_reward = -1e8 
for n_iter in range(MAX_ITER):
    st = time.time()
    RANDOM_SEED = n_iter
    """ Env and agent reset
    """
    day = day_shuffle[n_iter]
    env.reset(RANDOM_SEED, day)  # generate all day supplies and demands
    agent.reset_agent()
    fee_costs,profits,revenues,time_costs = [],[],[],[]
    losses_critic,losses_actor,rec_rewards = [],[],[]
    count_loss, count_query, count_service, count_success_service = 0, 0, 0, 0
    for cur_t in range(0, args.T_LEN):
        fee_cost, profit, revenue, cost, loss_critic, loss_actor, n_query, n_service, n_success_service, rec_reward = agent.step(cur_t, n_iter, is_val=False)
        count_loss += len(loss_critic)
        count_query += n_query
        count_service += n_service
        count_success_service += n_success_service
        fee_costs.extend(fee_cost)
        profits.extend(profit)
        revenues.extend(revenue)
        time_costs.extend(cost)
        rec_rewards.extend(rec_reward)
        losses_critic.extend(loss_critic)
        losses_actor.extend(loss_actor)

    mean_fees = round(np.mean(fee_costs),3)
    sum_profits = round(np.sum(profits),2)
    sum_revenues = round(np.sum(revenues),2)
    mean_time_costs = round(np.mean(time_costs),2)
    mean_reward = round(np.mean(rec_rewards),3)
    failure_rate = 1 - round(count_success_service/(count_service+args.eps),3)
    mean_losses_critic = np.mean(losses_critic)
    mean_losses_actor = np.mean(losses_actor)

    state = {'actor':agent.Actor.state_dict(), 
            'critic':agent.Critic.state_dict(),  
            'graphpool':agent.Graphpool.state_dict(), 
            'previous_rep':agent.previous_rep,
            'mean_fee':mean_fees,
            "profit":sum_profits,
            'time_cost':mean_time_costs,
            'failure_rate':failure_rate,
            'success_service_num':count_success_service,
            'mean_reward': mean_reward,
            'n_iter':n_iter,
            }
    if(not os.path.exists("./params")):
        os.mkdir("./params")
    torch.save(state, 'params/{}_{}.pkl'.format(file_name,n_iter))

    print("n_iter: {}".format(n_iter))
    logging.info("n_iter: {}".format(n_iter))
    print("Date: {}".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
    logging.info("Date: {}".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
    print("Training - n_query, n_service, n_suc_service, n_update: {},{},{},{}".format(count_query-1, count_service, count_success_service, count_loss))
    logging.info("Training - n_query, n_service, n_suc_service, n_update: {},{},{},{}".format(count_query-1, count_service, count_success_service, count_loss))
    print("profit, revenue, n_suc_service, fee, time_cost, failure_rate, reward: {}, {}, {}, {}, {}, {:.3}, {}".format\
        (sum_profits, sum_revenues, count_success_service, mean_fees, mean_time_costs, failure_rate, mean_reward))
    logging.info("profit, revenue, n_suc_service, fee, time_cost, failure_rate, reward: {}, {}, {}, {}, {}, {:.3}, {}".format\
        (sum_profits, sum_revenues, count_success_service, mean_fees, mean_time_costs, failure_rate, mean_reward))
    print("loss_critic,loss_actor: {:.3},{:.3}".format(mean_losses_critic, mean_losses_actor))
    logging.info("loss_critic,loss_actor: {:.3},{:.3}".format(mean_losses_critic, mean_losses_actor))
    print("time_consuming: {}s".format(int(time.time()-st)))
    logging.info("time_consuming: {}s".format(int(time.time()-st)))

    # previous_rep bkp 
    agent.previous_rep_train = [agent.previous_rep[0].clone(), agent.previous_rep[1].clone()]
    
    ### evaluation ###
    """ Env and agent reset
    """
    st = time.time()
    count_query, count_service, count_success_service = 0, 0, 0
    fee_costs,profits,revenues,time_costs,rec_rewards = [],[],[],[],[]
    days_test = [28]
    for d_test in days_test:
        env.reset(0, d_test) # generate all day supplies and demands
        agent.reset_agent()
        for cur_t in range(0,args.T_LEN):
            with torch.no_grad():
                fee_cost, profit, revenue, cost, _, _, n_query, n_service, n_success_service, rec_reward = agent.step(cur_t, n_iter, is_val=True)
            count_query += n_query
            count_service += n_service
            count_success_service += n_success_service
            fee_costs.extend(fee_cost)
            profits.extend(profit)
            revenues.extend(revenue)
            time_costs.extend(cost)
            rec_rewards.extend(rec_reward)

    mean_fees = round(np.mean(fee_costs),3)
    sum_profits = round(np.sum(profits),2)
    sum_revenues = round(np.sum(revenues),2)
    mean_time_costs = round(np.mean(time_costs),2)
    mean_reward = round(np.mean(rec_rewards),3)
    failure_rate = 1 - round(count_success_service/(count_service+args.eps),3)

    if(mean_reward>max_reward): 
        best_iter = n_iter
        best_fee = mean_fees
        best_profit = sum_profits
        best_revenue = sum_revenues
        best_timecost = mean_time_costs
        best_cfr = 1 - count_success_service/(count_service+args.eps)
        best_service_count = count_service
        best_sc_count = count_success_service
        max_reward = mean_reward

    print("Evaluation - n_query, n_service, n_suc_service: {},{},{}".format\
            (count_query-len(days_test), count_service, count_success_service))
    logging.info("Evaluation - n_query, n_service, n_suc_service: {},{},{}".format\
            (count_query-len(days_test), count_service, count_success_service))
    print("profit, revenue, n_suc_service, fee, time_cost, failure_rate, reward: {}, {}, {}, {}, {}, {:.3}, {}".format\
            (sum_profits, sum_revenues, count_success_service, mean_fees, mean_time_costs, failure_rate, mean_reward))
    logging.info("profit, revenue, n_suc_service, fee, time_cost, failure_rate, reward: {}, {}, {}, {}, {}, {:.3}, {}".format\
            (sum_profits, sum_revenues, count_success_service, mean_fees, mean_time_costs, failure_rate, mean_reward))
    val_metrics = {'profit': sum_profits/(50*4000),
               'revenue': sum_revenues/(50*4000),
               'n_sc_serve': count_success_service/1000,
               'mcp': mean_fees,
               'mcwt': mean_time_costs/10,
               'cfr': failure_rate,
               'reward': mean_reward
            }
    writer.add_scalars('val_metrics', val_metrics, n_iter)
    writer.flush()
    print("best_iter, profit, revenue, n_suc_service, n_service, fee, time_cost, failure_rate, max_reward: {}_iter, {}, {}, {}, {}, {}, {}, {:.3}, {}".format(best_iter, best_profit, best_revenue, best_sc_count, best_service_count, best_fee, best_timecost, best_cfr, max_reward))
    logging.info("best_iter, profit, revenue, n_suc_service, n_service, fee, time_cost, failure_rate, max_reward: {}_iter, {}, {}, {}, {}, {}, {}, {:.3}, {}".format(best_iter, best_profit, best_revenue, best_sc_count, best_service_count, best_fee, best_timecost, best_cfr, max_reward))
    print("time_consuming: {}s".format(int(time.time()-st)))
    logging.info("time_consuming: {}s".format(int(time.time()-st)))

    # previous_rep recover
    agent.previous_rep = [agent.previous_rep_train[0].clone(), agent.previous_rep_train[1].clone()]

writer.close()
    
