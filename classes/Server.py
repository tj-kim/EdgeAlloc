import numpy as np
import pdb


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

    def serve_users(self, usr_list, mu_bar = None, move_probs = None, rsv_flag = False, rsv_dynamic = False):
        
        load = len(usr_list)
        self.load_history += [load]
        
        num_users = self.mu.shape[0]
        overflow_flag = np.zeros(num_users)
        waittimes = np.zeros(num_users)
        
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
            
        if rsv_flag:
            if rsv_dynamic:
                waittimes = self.coordinate_users_dynamic(usr_list, mu_bar, move_probs)
            else:            
                waittimes = self.coorindate_users(usr_list, mu_bar, move_probs)
        
        return awarded, rewards, overflow_flag, waittimes
    
    def coorindate_users(self, usr_list, mu_bar, move_probs):
        
        # Find lowest n users and ban them for some time
        num_users = self.mu.shape[0]
        waittimes = np.zeros(num_users)
        num_kick = int(max(0, len(usr_list) - self.cap))
        compare_list = np.zeros(len(usr_list))
        
        if num_kick > 0:
            for i in range(len(usr_list)):
                compare_list[i] = mu_bar[i] * move_probs[i]
            
            # decide who to ban 
            idx_kick = np.argpartition(compare_list, num_kick)[:num_kick]
            idx_kick_sort = idx_kick[np.argsort(compare_list[idx_kick])]
            
            # decide on how long to ban
            idx_good = np.argpartition(compare_list, - self.cap)[-self.cap:]
            idx_good_sorted = idx_good[np.argsort(-compare_list[idx_good])]
            
            p_sub = np.array(move_probs)[idx_good_sorted]
            waittime = np.ceil((1 - np.prod(p_sub))**(-1))
            
            waittimes[np.array(usr_list)[idx_kick_sort]] = waittime
        
        return waittimes
    
    def coordinate_users_dynamic(self, usr_list, mu_bar, move_probs):
        
        # Find lowest n users and ban them for some time
        num_users = self.mu.shape[0]
        num_svrs = self.mu.shape[1]
        waittimes = np.zeros(num_users)
        num_kick = int(max(0, len(usr_list) - self.cap))
        compare_list = np.zeros(len(usr_list))
        
        if num_kick > 0:
            for i in range(len(usr_list)):
                compare_list[i] = mu_bar[i] # * move_probs[i]
            
            # decide who to ban 
            idx_kick = np.argpartition(compare_list, num_kick)[:num_kick]
            idx_kick_sort = idx_kick[np.argsort(compare_list[idx_kick])]
            
            # decide on how long to ban
            idx_good = np.argpartition(compare_list, - self.cap)[-self.cap:]
            idx_good_sorted = idx_good[np.argsort(-compare_list[idx_good])]
            
            pdb.set_trace()
            
            p_sub = np.array(move_probs)[idx_good_sorted]
            from_svr = (1 - np.prod(p_sub))
            
            p_sub_queue = np.array(move_probs)[idx_kick_sort]
            A = 0
            
            for uk in idx_kick_sort:
                
                from_queue = (np.prod(1 - p_sub_queue[:uk]))
                from_outside = (1 - (1-mu_bar[uk]) (num_users - C - A)/num_svrs)
                waittimes[np.array(usr_list)[uk]] = (from_svr * from_queue * from_outside)**(-1)
                
                A += 1
            
        
        return waittimes