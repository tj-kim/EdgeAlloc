import numpy as np

class Server():
    def __init__(self, capacity, s_idx, mu, location):
        
        self.cap = int(capacity)
        self.s_idx = s_idx
        self.location = location
        self.mu = mu
        self.load_history = []
        
    def receive_users(self, usr_list, mu_est_list, coordinate = False):        
    
        awarded, rewards, overflow_flag = self.serve_users(usr_list)
        if coordinate:
            wait_times = coordinate_users(self, mu_est_list, awarded, overflow_flag)
        else:
            wait_times = np.zeros(self.mu.shape[0])

        return awarded, rewards, wait_times

    def serve_users(self, usr_list):
        
        load = len(usr_list)
        self.load_history += [load]
        
        num_users = self.mu.shape[0]
        overflow_flag = np.zeros(num_users)
        
        # Select C users at random
        if load > self.cap:
            select_u = np.random.choice(usr_list, size=self.cap, replace=False)
            overflow_flag = np.ones(num_users)
        else:
            select_u = np.array(usr_list)
            
        rewards = np.zeros(self.mu.shape[0])
        awarded = np.zeros(self.mu.shape[0])
        for u in select_u:
            awarded[u] = True
        for u in usr_list:
            rewards[u] = int(np.random.rand() < self.mu[u, self.s_idx])
        
        return awarded, rewards, overflow_flag
    
    def coorindate_users(self, mu_est_list, awarded, overflow_flag):
        
        # Find lowest n users and ban them for some time
        
        
        return