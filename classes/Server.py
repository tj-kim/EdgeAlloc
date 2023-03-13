class Server():
    def __init__(self, capacity, s_idx, mu, location):
        
        self.cap = capacity
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
        
        overflow_flag = False
        
        # Select C users at random
        if load > self.capacity:
            select_u = np.random.choice(usr_list, size=self.cap, replace=False)
            overflow_flag = True
        else:
            select_u = np.array(usr_list)
            
        rewards = np.zeros(self.mu.shape[0])
        awarded = np.zeros(self.mu.shape[0])
        for u in select_u:
            awarded[u] = True
            rewards[u] = int(np.random.rand() < self.mu[u, s_idx])
        
        return awarded, rewards, overflow_flag
    
    def coorindate_users(self, mu_est_list, awarded, overflow_flag):
        
        # Find lowest n users and ban them for some time
        
        
        return