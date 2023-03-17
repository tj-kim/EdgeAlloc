import numpy as np
import itertools
from tools.solver_utils import *

import pdb

def gen_eq_locs(space_1d, nums, offset = 0.5):
    # Generate well spread out locations in square space
    num_across = int(np.floor(np.sqrt(nums)))
    locs = []
    
    inc = space_1d/num_across
    
    for i,j in itertools.product(range(num_across), range(num_across)):
        locs += [(i*inc+offset, j*inc+offset)]
    
    return locs

def dist_usr_arms(users):
    # select arms for all users and return vector of server to arm 
    
    lrn_x = np.zeros([len(users), users[0].num_servers])
    
    for u in range(len(users)):
        arm_id = users[u].select_arm_UCB2()
        lrn_x[u,arm_id] = 1
    
    return lrn_x

def dist_receive_rewards(servers, users, lrn_x_dist):
    # Obtain rewards and update user UCB records from servers
    
    # Per server accept list
    usr_list_dict = {}
    mu_list_dict = {}
    
    # Organize user list for each server
    for s in range(len(servers)):
        usr_list_dict[s] = []
        for u in range(len(users)):
            if lrn_x_dist[u,s] > 0:
                usr_list_dict[s] += [u]

    # Let each server process user
    awarded = np.zeros_like(servers[0].mu)
    rewards = np.zeros_like(servers[0].mu)
    overflow_flag = np.zeros_like(servers[0].mu)
    
    for s in range(len(servers)):
        awarded[:,s], rewards[:,s], overflow_flag[:,s] = servers[s].serve_users(usr_list_dict[s])
        
    # Update each user's history based on server response
    for u in range(len(users)):
        arm_id = np.argmax(lrn_x_dist[u])
        reward = rewards[u,arm_id]
        users[u].receive_reward(arm_id, reward)
    
    return

def characterize_collision(x,B,C):
    
    U, K = x.shape
    loss_val = 0
    collision_rate = 0
    
    for u in range(U):
        for k in range(K):
            denom = 1e-5
            for u2 in range(U):
                denom += x[u2,k]
            collision_rate += (denom/U) * (1 - min(C[k]/denom, 1))
            loss_val += x[u,k] * B[u,k] * (1 - min(C[k]/denom, 1))
    return loss_val, collision_rate