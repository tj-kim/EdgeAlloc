import numpy as np


def update_L_random(U,K, num_select):
    # Randomly generate available arms for each user 
    
    L = np.zeros([U,K])
    for u in range(U):
        idx = np.random.choice(range(K), size=num_select, replace=False, p=None)
        for i in idx:
            L[u,i] = 1
    
    return L

def obtain_L_users(U, K, users):
    # Pull arm availability from user instance based on user's current loc
    
    L = np.zeros([U,K])
    for u in range(U):
        valid_list = users[u].arms_per_loc[users[u].usr_place]
        for j in valid_list:
            L[u,j] = 1
            
    return L