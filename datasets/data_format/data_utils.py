
import numpy as np
import torch
import random
import os

import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from statsmodels.graphics.tsaplots import plot_acf

sns.set(style='ticks', font_scale=1.2)

def get_phi(x):
    all_ = []
    
    for i, entity in enumerate(x):
        all_.append((x[entity].max(axis = 0) - x[entity].min(axis = 0)).reshape(1,-1))
    return (np.concatenate(all_, axis = 0)).mean(axis = 0)

class Standardization:
    # x and y are dictionary
    def __init__(self, x, appr_healthy_state = 30):
        self.mean = None
        self.std = None
        self.appr_healthy_state = appr_healthy_state
        self.fit(x)
    def fit(self, x):
        x_ = []
        for i in x:
            x_.append(x[i][:self.appr_healthy_state,:]) # consider the time steps upto 30 as healthy state
        x_ = np.concatenate(x_, axis = 0)
        self.mean = x_.mean(axis = 0, keepdims = True)
        self.std = x_.std(axis = 0, keepdims = True)
    def transform(self, y):
        for i, j in enumerate (y):
            y[j] = (y[j] - self.mean) / self.std
        return y
    def get(self):
        return self.mean, self.std

def even_chunks(n):
    assert n % 2 == 0
    # Generate n evenly spaced points between -1 and 1
    points = np.linspace(-1, 1, num=n+1)
    # Create chunks based on the generated points
    chunks = []
    for i in range(n):
        lower = points[i]
        upper = points[i+1]
        chunks.append((lower, upper))
    return chunks 

def get_continuous_property(X, stds_ = None,
                            varying = True, swapping_map = None):
    # get total num of observations  
    num_obs = 0
    for xx in (X.values()):
        num_obs += xx.shape[0]
    shape_ = (num_obs, xx.shape[-1])
    
    # We generate continuous property same along the feature dim 
    len_, channel_dim = shape_

    # Create synthetic continous property φc_t. 
    if stds_ is None:
        # hard-coded coefficients, which is much higher variance than (max-min) of signal
       phi_ = np.array([1., 2., 3.5, 2.7, 2.0, 3.0, 0.1, 2.2, 1.6, 2.2, 0.1, 1.1, 0.15, 0.3]) * 30 
    else: # based on min-max variation
        phi_ = stds_ * 2
    # Generate c for all observations 
    continuous_ocs = (np.random.rand(len_) * 2) -1 # [-1,1]
    continuous_ocs = np.expand_dims(continuous_ocs,1)
    
    
    '''
    Oerating condition for each observation at time t is the same across its channels.
    But, how the same operating condition affects each channel is different
    We do this by randomly permuting chunks of values in the range [-1,1] across the channels
    '''
    continuous_ocs = np.repeat(continuous_ocs,channel_dim,1)

    # We set even number of reference chunks and pair them randomly to get a swapping map for each feature
    if varying:
        if swapping_map is None:
            d_ = True
            chunks_num = 8
            iii = 0
            while d_:
                chunks = even_chunks(chunks_num)
                # Randomly pairing the chunks for each feature and transform
                swapping_map = [] # this must be the same for training and testing data
                for j in range (1,channel_dim):
                    random.shuffle(chunks)
                    random_pairs = [(chunks[i], chunks[i+1]) for i in range(0,len(chunks), 2)]
                    swapping_map.append(random_pairs) # the reference column is not included
                    # for each pair, do swap
                    for pair in random_pairs:
                        range_a, range_b = pair
                        distance = range_b[0] - range_a[0] # (+)a->b  (-) a<-b
                        # indices
                        a_to_b = np.logical_and(continuous_ocs[:, 0] >= range_a[0], continuous_ocs[:, 0] < range_a[1])
                        b_to_a = np.logical_and(continuous_ocs[:, 0] >= range_b[0], continuous_ocs[:, 0] < range_b[1])
                        # swap
                        continuous_ocs[a_to_b, j] = continuous_ocs[a_to_b, 0] + distance
                        continuous_ocs[b_to_a, j] = continuous_ocs[b_to_a, 0] - distance
                # To check 
                check_ = np.all(continuous_ocs[:, np.newaxis, :] == continuous_ocs[:, :, np.newaxis], axis=0)
                d_ = np.any(check_ & ~np.eye(check_.shape[0], dtype=bool))
                iii += 1
                if iii > 6: # if the loop takes too long increases the chunk number
                    chunks_num += 2
        else:
            for j in range (1,channel_dim):
                random_pairs = swapping_map[j-1]
                # for each pair, do swap
                for pair in random_pairs:
                    range_a, range_b = pair
                    distance = range_b[0] - range_a[0] # (+)a->b  (-) a<-b
                    # indices
                    a_to_b = np.logical_and(continuous_ocs[:, 0] >= range_a[0], continuous_ocs[:, 0] < range_a[1])
                    b_to_a = np.logical_and(continuous_ocs[:, 0] >= range_b[0], continuous_ocs[:, 0] < range_b[1])
                    # swap
                    continuous_ocs[a_to_b, j] = continuous_ocs[a_to_b, 0] + distance
                    continuous_ocs[b_to_a, j] = continuous_ocs[b_to_a, 0] - distance
    
    # use base c_t as ground truth continuous ocs
    gt_ocs = continuous_ocs[:,0:1]
    # φc_t
    continuous_ocs = continuous_ocs * phi_.reshape(1,-1)
    # print(phi_)
    
    gt_c = dict()
    idx_from = 0
    for xx in (X):
        idx_to = X[xx].shape[0] + idx_from
        X[xx] = X[xx] + continuous_ocs[idx_from: idx_to, :]
        
        gt_c[xx] = gt_ocs[idx_from: idx_to, :]
    return X, swapping_map, gt_c

def sliding_window(X, ocs, rul, w_T, 
                   H = 3, H_lookback = 1, trajectory = None):
    keys = X.keys()
    windowed_X, OCS, RUL = [], [], []
    
    time_indices = []
    
    # Hypers
    windowed_X_lookback = []
    windowed_X_horizon = []
    windowed_OCS_lookback = []
    windowed_OCS_horizon = []
    windowed_tidx_lookback = []
    for key in keys:
        seq_X, seq_ocs, rul_key = X[key], ocs[key], rul[key]  # (L_k, c_x), (L_k, c_ocs), (L_k, c_z)
        L_k = len(seq_X)
        
        # Extract channel dimensions
        c_x, c_ocs = seq_X.shape[1], seq_ocs.shape[1]
        
        # Left padding with first observation + noise
        pad_x = np.tile(seq_X[0], (w_T - 1, 1)) + np.random.normal(0, 0.02, (w_T - 1, c_x))
        pad_ocs = np.tile(seq_ocs[0], (w_T - 1, 1))
        
        # Padded sequences
        padded_X = np.vstack([pad_x, seq_X])
        padded_OCS = np.vstack([pad_ocs, seq_ocs])
        
        # Apply sliding window
        windowed_x_k = np.lib.stride_tricks.sliding_window_view(padded_X, (w_T, c_x)).squeeze(1) # (num_w, leng_w, c)
        windowed_ocs_k = np.lib.stride_tricks.sliding_window_view(padded_OCS, (w_T, c_ocs)).squeeze(1)
        
        # Generate time indices
        time_index_k = np.arange(1, L_k + 1).reshape(-1, 1)
        
        if trajectory:
            hyperW_X_lookback, hyperW_X_horizon, hyperW_ocs_lb, hyperW_ocs_horizon, hyperW_tidx  = \
                                                    hyper_sliding_window(windowed_x_k, windowed_ocs_k, time_index_k,
                                                               H, H_lookback, w_T, trajectory)
            windowed_X_lookback.append(hyperW_X_lookback)
            windowed_X_horizon.append(hyperW_X_horizon)
            windowed_OCS_lookback.append(hyperW_ocs_lb)
            windowed_OCS_horizon.append(hyperW_ocs_horizon)
            windowed_tidx_lookback.append(hyperW_tidx)
            
        # else:
        #     # Store results
        windowed_X.append(windowed_x_k)
        OCS.append(windowed_ocs_k)
        RUL.append(rul_key)
        time_indices.append(time_index_k)
        
    # Concatenate all keys together
    windowed_X = np.concatenate(windowed_X, axis=0)
    OCS = np.concatenate(OCS, axis=0)
    RUL = np.concatenate(RUL, axis=0)
    time_indices = np.concatenate(time_indices, axis=0)
    
    RL_rul_sets = {"X": torch.tensor(windowed_X, dtype=torch.float32),
                "ocs": torch.tensor(OCS, dtype=torch.float32),
                "RUL": torch.tensor(RUL, dtype=torch.float32),
                "t_idx": torch.tensor(time_indices, dtype=torch.float32)}
    
    if trajectory:
        windowed_X_lookback = np.concatenate(windowed_X_lookback, axis=0)
        windowed_X_horizon = np.concatenate(windowed_X_horizon, axis=0)
        windowed_OCS_lookback = np.concatenate(windowed_OCS_lookback, axis=0)
        windowed_OCS_horizon = np.concatenate(windowed_OCS_horizon, axis=0)
        windowed_tidx_lookback = np.concatenate(windowed_tidx_lookback, axis=0)
        trajectory_sets = {"X_lookback": torch.tensor(windowed_X_lookback, dtype=torch.float32),
                           "ocs_lookback": torch.tensor(windowed_OCS_lookback, dtype=torch.float32),
                           "t_idx_lookback": torch.tensor(windowed_tidx_lookback, dtype=torch.float32),
                           "X_horizon": torch.tensor(windowed_X_horizon, dtype=torch.float32),
                           "ocs_horizon": torch.tensor(windowed_OCS_horizon, dtype=torch.float32)}
    else:
        trajectory_sets = None
    return RL_rul_sets, trajectory_sets

def hyper_sliding_window(windowed_X, windowed_ocs, time_indices,
                         H, H_lookback, T, trajectory):
    # TODO consider feature-based trj and data-based trj 
    hyperW_X_lookback, hyperW_X_horizon = [], [] 
    hyperW_OCS_lookback, hyperW_OCS_horizon, hyperW_timeidx = [], [], []
    
    # for windowed_X in WX:

    L, c, d = windowed_X.shape
    if trajectory == "feature":
        valid_L = L - H  # The index where both B and H windows are valid
        W_lookback = np.lib.stride_tricks.sliding_window_view(windowed_X[:valid_L], (H_lookback, c, d))[:, :, 0]
        W_lookback = np.squeeze(W_lookback, axis=1)
        W_H = np.lib.stride_tricks.sliding_window_view(windowed_X[H_lookback:valid_L+H_lookback], (H, c, d))[:, :, 0]
        W_H = np.squeeze(W_H, axis=1)
        
        ocs_lookback = np.lib.stride_tricks.sliding_window_view(windowed_ocs[:valid_L], (H_lookback, c, d))[:, :, 0]
        ocs_lookback = np.squeeze(ocs_lookback, axis=1)
        timeidx_lookback = np.lib.stride_tricks.sliding_window_view(time_indices[:valid_L], (H_lookback, 1))
        timeidx_lookback  = np.squeeze(timeidx_lookback , axis=1)
        
        hyperW_OCS_horizon = np.lib.stride_tricks.sliding_window_view(windowed_ocs[H_lookback:valid_L+H_lookback], (H, c, d))[:, :, 0]
        hyperW_OCS_horizon = np.squeeze(hyperW_OCS_horizon, axis=1)
        
        
        return W_lookback, W_H, ocs_lookback, hyperW_OCS_horizon, timeidx_lookback
        # print("!!", W_lookback)
        # hyperW_X_lookback.append((W_lookback))
        # hyperW_X_horizon.append((W_H))
        
    elif trajectory == "data":
        forward_frame_length = (H-1)*T + 1 + T
        backward_frame_length = (H_lookback-1)*T + 1
        valid_L = L - forward_frame_length  # The index where both B and H windows are valid
        if valid_L < backward_frame_length:
            raise ValueError("Not enough data points to create the desired sliding windows.")
        for l in range(valid_L):
            if l+1 < backward_frame_length:
                continue
            else: 
                hyperW_X_lookback.append(windowed_X[None, l+1- backward_frame_length:l+1:T,:,:])
                hyperW_X_horizon.append(windowed_X[None, l + T:l+forward_frame_length:T, :, :])
                hyperW_timeidx.append(time_indices[None, l+1- backward_frame_length:l+1:T,:])
                hyperW_OCS_lookback.append(windowed_ocs[None, l+1- backward_frame_length:l+1:T,:,:])
                hyperW_OCS_horizon.append(windowed_ocs[None, l + T:l+forward_frame_length:T, :, :])
                
        hyperW_X_lookback = np.concatenate((hyperW_X_lookback), axis = 0)
        hyperW_X_horizon = np.concatenate(hyperW_X_horizon, axis = 0)
        hyperW_OCS_lookback = np.concatenate((hyperW_OCS_lookback), axis = 0)
        hyperW_OCS_horizon = np.concatenate(hyperW_OCS_horizon, axis = 0)
        hyperW_timeidx = np.concatenate((hyperW_timeidx), axis = 0)

        # return hyperW_X_lookback, hyperW_X_horizon
        return hyperW_X_lookback, hyperW_X_horizon, \
            hyperW_OCS_lookback, hyperW_OCS_horizon, hyperW_timeidx

def parse_lr_set(windowed_set):
    return windowed_set["X"], windowed_set["ocs"], windowed_set["RUL"], windowed_set["t_idx"]

def parse_trj_set(windowed_set):
    return ([windowed_set["X_lookback"], windowed_set["ocs_lookback"], windowed_set["t_idx_lookback"]], 
            [windowed_set["X_horizon"], windowed_set["ocs_horizon"]])


def plot_data(x, inst = 1, path_ = None):
    N = len(x)
    if inst > N:
        raise ValueError("S must be less than or equal to N to ensure unique values.")
    indx = random.sample(range(1, N + 1), inst)
    
    # x is 2d array
    X = x
    
    for i in indx:
        X = x[i]
        for c in range(X.shape[1]):
            fig = plt.figure(figsize=(12, 8), 
                            dpi = 600) 
            axes = fig.subplots()
            axes.plot(X[:,c], c = "blue", linewidth=1, alpha=1)
            # plt.title(title_)
            plt.xlabel("Time")
            plt.ylabel("Magnitude")
            plt.xticks(fontweight='bold', fontsize = 15)   
            plt.yticks(fontweight='bold', fontsize = 15)
            plt.tight_layout()
            if path_ :
                plt.savefig(os.path.join(path_+ f"machine{i}_channel_{c}.png" )) 
            else:
                plt.show()    
            plt.clf()   
            plt.close(fig)
            ACF_(X[:,c], i = i, c = c, path_ = path_)

def ACF_(x_, lag = 30, i = 1, c = 1,
         path_ = None):
    '''
    This function is to examine auto-correlation of the input time series x
    x is in shape (sequence len, feature)
    '''
    # x_ is input NS time series
    fig = plt.figure(figsize=(12, 8), 
            dpi = 600) 
    axes = fig.subplots()
    plot_acf(x_, lags = lag, title = '')
    plt.xlabel("Lag",fontsize= 15,weight = "bold")
    plt.ylabel("Pearson’s correlation coefficient",fontsize= 15,weight = "bold")
    plt.xticks(fontweight='bold', fontsize= 15)   
    plt.yticks(fontweight='bold', fontsize= 15)
    plt.tight_layout()
    if path_:
        plt.savefig(os.path.join(path_+ f"machine{i}_channel_{c}.png" )) 
    else:    
        plt.show()
    plt.clf()   
    plt.close(fig)
    
    