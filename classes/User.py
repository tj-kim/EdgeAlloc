import numpy as np
import itertools
import random
import copy

# from classes.distributed_utils import *

class User():

    def __init__(self, servers, T, locs, threshold_dist = 6, lat_dist = 4, self_weight = 0.5):
        
        self.num_servers = len(servers)        
        self.pulls, self.means = np.ones(self.num_servers + 1), np.ones(self.num_servers + 1)
        self.t = int(self.num_servers)
        self.UCB = self.update_UCB()
        self.reward_log = np.zeros(T)
        self.arm_history = np.zeros(T)
        self.wait_times = np.zeros(self.num_servers)
        
        self.locs = locs
        self.dists = self.get_dists()
        
        self.svr_locs = self.get_svr_locs(servers)
        self.usr_place = np.random.randint(len(locs))
        
        self.P_loc = None
        self.gen_MC(threshold_dist, self_weight)
        
        self.dist2_svr = self.get_loc2serv_dist()
        self.arms_per_loc = self.get_arms_per_loc(lat_dist)
        
    def gen_MC(self, threshold_dist = 6, self_weight = 0.5):
        # Creating Markov Transition Probability Matrix 
        
        P = np.zeros(self.dists.shape)
        locs = self.locs
        for i in range(len(locs)):
            cut_list = self.dists[i,:]
            others = np.squeeze(np.argwhere((cut_list > 0) * (cut_list < threshold_dist) == True))
            num_others = others.shape[0]
        
            # Draw values to make up row of MC
            self_transition = np.random.exponential(scale=1/self_weight)
            others_transition = np.random.exponential(scale=1/((1-self_weight)*num_others),size=num_others)
            total = self_transition + np.sum(others_transition)
            
            P[i,i] = self_transition/total
            
            idx = 0
            for j in others:
                P[i,j] = others_transition[idx]/total
                idx += 1
            
        self.P_loc = P
        self.next_loc()
        
    def get_dists(self):
        # Obtaining distance matrix (from loc to loc) 
        
        locs = self.locs
        
        num_locs = len(locs)
        dists = np.zeros([num_locs,num_locs])
        
        for i,j in itertools.product(range(num_locs), range(num_locs)):
            if dists[i,j] == 0 and i != j:
                a = np.array(locs[i])
                b = np.array(locs[j])
                dists[i,j] = np.linalg.norm(a-b)
                dists[j,i] = dists[i,j]
        
        return dists
    
    def next_loc(self):
        # Update user location based on markov chain
        weights = self.P_loc[self.usr_place]
        population = range(weights.shape[0])
        self.usr_place =  random.choices(population, weights)[0]
    
    def get_svr_locs(self, servers):
        server_locs = []
        
        for s in range(len(servers)):
            server_locs += [servers[s].location]
            
        return server_locs
    
    def get_arms_per_loc(self, lat_dist):
        
        svr_avail = {}
        
        # Get available arms for each markov chain location
        for i in range(len(self.locs)):
            candidates = []
            dists = self.dist2_svr[i]
            for s in range(len(dists)):
                dists_s = dists[s]
                if dists[s] <= lat_dist:
                    candidates += [s]
            svr_avail[i] = candidates
            
        return svr_avail
    
    def get_loc2serv_dist(self):
        
        # Getting distance from each user location to each server
        temp_loc_dists = {}
        
        locs = self.locs
        svr_locs = self.svr_locs
        idx = 0
        
        for u_loc in locs:
            temp_u_loc = []
            
            for s_loc in svr_locs:
                a = np.array(u_loc)
                b = np.array(s_loc)
                temp_u_loc += [np.linalg.norm(a-b)]
            
            temp_loc_dists[idx] = temp_u_loc
            idx += 1
                    
        return temp_loc_dists
        
        # Changing latency to threshold
        temp_loc_thresh = []
        
        for u in range(len(locs)):
            temp_u_thresh = []
            
            for s in range(len(svr_locs)):
                temp_u_thresh += [max(self.latency_threshold - temp_loc_lats[u][s],0)]
            
            temp_loc_thresh += [temp_u_thresh]
         
        self.loc_thresh = temp_loc_thresh
        
    
    def select_arm_random(self):
        # Amongst the available arms
        arm_id = np.random.choice(arms_per_loc[self.usr_place])
        
        if arm_id is None:
            arm_id = self.num_servers
            
        self.arm_history[self.t] = int(arm_id)
        
        return arm_id
    
    def select_arm_UCB2(self): 
        
        # Filter out unavailable arms
        UCB = copy.deepcopy(self.UCB)
        for i in range(self.num_servers): # Exclude dummy server
            if i not in self.arms_per_loc[self.usr_place] or self.wait_times[i] > 0:
                UCB[i] = -1
        
        arm_id = np.random.choice(np.flatnonzero(UCB == UCB.max()))
        self.arm_history[self.t] = int(arm_id)
        return arm_id

    def update_UCB(self):
        
        UCB_temp = np.zeros(self.num_servers + 1)
        K = self.num_servers
        
        for k in range(K):
            UCB_temp[k] = self.means[k] + np.sqrt(2 * np.log(self.t) / self.pulls[k])
            
        return UCB_temp
        
    def receive_reward(self, arm_id, reward, waittime = 0):
    
        self.means[arm_id] = (self.means[arm_id] * self.pulls[arm_id] + reward) / (self.pulls[arm_id] + 1)
        self.pulls[arm_id] += 1

        self.UCB = self.update_UCB()
        
        return
    
        
    def next_step(self):
        self.next_loc()
        self.t += 1
        
        self.wait_times -= np.ones_like(self.wait_times)
        self.wait_times = np.maximum(self.wait_times, 0)
    