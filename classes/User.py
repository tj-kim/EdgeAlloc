import numpy as np

import pickle
import os
import numpy as np
import itertools
import random


from classes.distributed_utils import *

class User():

    def __init__(self, servers, T, locs, max_dist = 7, threshold_dist = 6, self_weight = 0.5):
                
        self.pulls, self.means, self.ucb_idx = np.zeros(num_servers), np.zeros(num_servers), np.zeros(num_servers)
        self.t = int(0)
        self.reward_log = np.zeros(T)
        self.arm_history = np.zeros(T)
        self.num_servers = len(servers)
        self.latency_conversion = latency_conversion # converts distance to time 
        
        self.locs = locs
        self.dists = self.get_dists()
        
        self.svr_locs = self.get_svr_locs(servers)
        self.usr_place = np.random.randint(len(locs))
        
        self.P_loc = None
        self.gen_MC(max_dist, threshold_dist, self_weight)
        
        self.loc_dists = None
        self.loc_lats = None
        self.loc_thresh = None 
        self.get_loc2serv_dist()
        
    def gen_MC(self, max_dist = 7, threshold_dist = 6, self_weight = 0.5):
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
    
    def get_arms_per_loc(self):
        
        # Get available arms for each markov chain location
    
    def get_loc2serv_dist(self):
        
        # Getting distance from each user location to each server
        temp_loc_dists = []
        
        locs = self.locs
        svr_locs = self.svr_locs
        
        for u_loc in locs:
            temp_u_loc = []
            
            for s_loc in svr_locs:
                a = np.array(u_loc)
                b = np.array(s_loc)
                temp_u_loc += [np.linalg.norm(a-b)]
            
            temp_loc_dists += [temp_u_loc]
            
        self.loc_dists = temp_loc_dists
        
        # changing distance to latency
        temp_loc_lats = []
        
        for u in range(len(locs)):
            temp_u_lat = []
            
            for s in range(len(svr_locs)):
                temp_u_lat += [temp_loc_dists[u][s] * self.latency_conversion]
                
            temp_loc_lats += [temp_u_lat]
            
        self.loc_lats = temp_loc_lats
        
        # Changing latency to threshold
        temp_loc_thresh = []
        
        for u in range(len(locs)):
            temp_u_thresh = []
            
            for s in range(len(svr_locs)):
                temp_u_thresh += [max(self.latency_threshold - temp_loc_lats[u][s],0)]
            
            temp_loc_thresh += [temp_u_thresh]
         
        self.loc_thresh = temp_loc_thresh
            
    def select_arm_closest(self):
        # Baseline algorithm
        temp_max = np.array(self.loc_dists[self.usr_place]).min()
        arm_id = np.random.choice(np.flatnonzero(self.loc_dists[self.usr_place] == temp_max))
        self.arm_history[self.t] = int(arm_id)
    
        return arm_id
    
    def select_arm_random(self):
        arm_id = np.random.choice(range(self.num_servers))
        self.arm_history[self.t] = int(arm_id)
        
        return arm_id
    
    def select_arm_dist(self, server_dists, server_rates): # Assuming linear, dists takes Vs value
        
        prob_lists = np.zeros(self.num_servers)
        
        for i in range(self.num_servers):
            
            lat = self.loc_thresh[self.usr_place][i]
            
            if lat > 0:
                L = server_dists[i]
                C = server_rates[i]
                prob_lists[i] = lat * (np.exp(-self.load * C/lat) - np.exp(-(self.load + L) * C/lat))/ (C * L)
            else:
                prob_lists[i] = 0
            
        arm_id = np.random.choice(np.flatnonzero(prob_lists == prob_lists.max()))
        self.arm_history[self.t] = int(arm_id)
        return arm_id

    
    def log_reward(self, latency):
        
        curr_reward = 0
        arm_id = int(self.arm_history[self.t])
        
        lat_compare = self.loc_thresh[self.usr_place][arm_id]
        if latency < lat_compare:
            curr_reward = 1

        self.reward_log[self.t] = curr_reward

        self.pulls[arm_id] += 1
        self.t += int(1)
        
        return curr_reward