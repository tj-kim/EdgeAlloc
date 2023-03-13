import numpy as np


def update_L_random(U,K, num_select):
    L = np.zeros([U,K])
    for u in range(U):
        idx = np.random.choice(range(K), size=num_select, replace=False, p=None)
        for i in idx:
            L[u,i] = 1
    
    return L