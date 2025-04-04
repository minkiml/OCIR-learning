
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
    
    def de_normalization(self, y):
        
        for i, j in enumerate (y):
            y[j] = y[j] * self.std + self.mean
        return y
    
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
                            varying = True, swapping_map = None,
                            add_cont = True, vis = False):
    # get total num of observations  
    num_obs = 0
    for xx in (X.values()):
        num_obs += xx.shape[0]
    shape_ = (num_obs, xx.shape[-1])
    
    # We generate continuous property same along the feature dim 
    len_, channel_dim = shape_

    # Create synthetic continous property φc_t. 
    if stds_ is None:
        # hard-coded coefficients
       phi_ = np.array([1., 2., 3.5, 2.7, 2.0, 3.0, 0.1, 2.2, 1.6, 2.2, 0.1, 1.1, 0.15, 0.3]) * 30 
    else: # based on min-max variation
        phi_ = stds_ * 2.0
    # Generate c for all observations 
    continuous_ocs = (np.random.rand(len_) * 2) -1 # [-1,1]
    # continuous_ocs = (np.random.randn(len_))  # N(0,1)
    continuous_ocs = np.expand_dims(continuous_ocs,1)
    continuous_ocs = np.repeat(continuous_ocs,channel_dim,1)
    
    '''
    Oerating condition for each observation at time t is the same across its channels.
    But, how the same operating condition affects each channel is different
    We do this by randomly permuting chunks of values in the range [-1,1] across the channels
    '''

    # Piece-wise function c_i = f_i(c_1) for i = 2, ..., dx
    if varying:
        if swapping_map is None:
            d_ = True
            chunks_num = 4
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
                d_ = False
                # d_ = np.any(check_ & ~np.eye(check_.shape[0], dtype=bool))
                # iii += 1
                # if iii > 6: # if the loop takes long increases the chunk number
                #     chunks_num += 2
            print("chunk number: ", chunks_num)
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
    if vis and varying:
        c1_values = np.linspace(-1, 1, 500)
        c1_values = np.expand_dims(c1_values,1)
        c1_values = np.repeat(c1_values,channel_dim,1)
        for j in range (1,channel_dim):
            random_pairs = swapping_map[j-1]
            # for each pair, do swap
            for pair in random_pairs:
                range_a, range_b = pair
                distance = range_b[0] - range_a[0] # (+)a->b  (-) a<-b
                # indices
                a_to_b = np.logical_and(c1_values[:, 0] >= range_a[0], c1_values[:, 0] < range_a[1])
                b_to_a = np.logical_and(c1_values[:, 0] >= range_b[0], c1_values[:, 0] < range_b[1])
                # swap
                c1_values[a_to_b, j] = c1_values[a_to_b, 0] + distance
                c1_values[b_to_a, j] = c1_values[b_to_a, 0] - distance
    else: c1_values = None
    # use base c_t as ground truth continuous ocs
    gt_ocs = continuous_ocs[:,0:1]
    # φc_t
    continuous_ocs = continuous_ocs * phi_.reshape(1,-1)
    # print(phi_)
    
    gt_c = dict()
    idx_from = 0
    for xx in (X):
        idx_to = X[xx].shape[0] + idx_from
        if add_cont:
            X[xx] = X[xx] + continuous_ocs[idx_from: idx_to, :]
        else: pass    
        gt_c[xx] = gt_ocs[idx_from: idx_to, :]
        
        idx_from = idx_to
    return X, swapping_map, gt_c, c1_values

def sw_function(x, ocs, w_T):
    seq_X, seq_ocs= x, ocs  # (L_k, c_x), (L_k, c_ocs), (L_k, c_z)
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
    return windowed_x_k, windowed_ocs_k, time_index_k

def sliding_window(X, ocs, rul, w_T, 
                   H = 3, H_lookback = 1, trajectory = None,
                   in_dictionary = False):
    keys = X.keys()
    windowed_X, OCS, RUL = [], [], []
    
    time_indices = []
    
    RL_rul_sets = dict()
    for key in keys:
        rul_key = rul[key]
        windowed_x_k, windowed_ocs_k, time_index_k = sw_function(X[key], ocs[key], w_T)
        
        if not in_dictionary:
            windowed_X.append(windowed_x_k)
            OCS.append(windowed_ocs_k)
            RUL.append(rul_key)
            time_indices.append(time_index_k)
        else:
            RL_rul_sets[key] = {"X": torch.tensor(windowed_x_k, dtype=torch.float32),
                                "ocs": torch.tensor(windowed_ocs_k, dtype=torch.float32),
                                "RUL": torch.tensor(rul_key, dtype=torch.float32),
                                "t_idx": torch.tensor(time_index_k, dtype=torch.long)}
    if not in_dictionary:
        # Concatenate all keys together
        windowed_X = np.concatenate(windowed_X, axis=0)
        OCS = np.concatenate(OCS, axis=0)
        RUL = np.concatenate(RUL, axis=0)
        time_indices = np.concatenate(time_indices, axis=0)
        
        RL_rul_sets = {"X": torch.tensor(windowed_X, dtype=torch.float32),
                    "ocs": torch.tensor(OCS, dtype=torch.float32),
                    "RUL": torch.tensor(RUL, dtype=torch.float32),
                    "t_idx": torch.tensor(time_indices, dtype=torch.long)}

    return RL_rul_sets
def apply_hyper_window(data_dict, H, W, S):
        """
        Apply hyper lookback and hyper horizon windows to the windowed sequences.
    
        Args:
            data_dict (dict): Dictionary with keys {1, 2, ..., K}, each containing:
                - "windowed_X": (N, T, dx)
                - "windowed_ocs": (N, T, dc)
                - "windowed_time": (N, 1)
            H (int): Size of hyper horizon window (future)
            W (int): Size of hyper lookback window (past)
            S (int): Stride size (ensures no overlap)
        
        Returns:
            hyper_windowed_dict (dict): Dictionary with the new hyper windowed values.
        """
        hyper_windowed_dict = {}
        hyper_lookback_X_all = []
        hyper_horizon_X_all = []
        hyper_lookback_ocs_all = []
        hyper_horizon_time_all = []
        for key, values in data_dict.items():
            windowed_X = values["X"]  # (N, T, dx)
            windowed_ocs = values["ocs"]  # (N, T, dc)
            windowed_time = values["t_idx"]  # (N, 1)
            
            N, T, dx = windowed_X.shape
            dc = windowed_ocs.shape[-1]
    
            # Compute the number of hyper windows M
            M = (N + S - 1) // S  # Equivalent to ceil(N / S)
            
            # Right pad `windowed_X` to ensure last window is included
            
            pad_size = (M * S) - N
            windowed_X_padded = np.pad(windowed_X, ((0, pad_size), (0, 0), (0, 0)), mode="edge")
            windowed_ocs_padded = np.pad(windowed_ocs, ((0, pad_size), (0, 0), (0, 0)), mode="edge")
            windowed_time_padded = np.pad(windowed_time, ((0, pad_size), (0, 0)), mode="edge")
            
            # print(windowed_X_padded.shape)
            # Collect hyper lookback and hyper horizon windows
            
            start_W = (W - 1) * S
            end_H = windowed_X_padded.shape[0]
            
            hyper_lookback_X = []
            hyper_horizon_X = []
            hyper_lookback_ocs = []
            hyper_horizon_time = []
             # TODO: we need to make lookback data ascending order
            for i in range(windowed_X_padded.shape[0]):
                if i >= start_W:
                    WX = np.array([windowed_X_padded[i-(S*j)] for j in range(W)]) # (W, T, dx)
                    WX = np.flip(WX, axis = 0).copy()
                    WOCS = np.array([windowed_ocs_padded[i-(S*j)] for j in range(W)])
                    WOCS = np.flip(WOCS, axis = 0).copy()
                    WTIDX = np.array([windowed_time_padded[i-(S*j)] for j in range(W)])
                    WTIDX = np.flip(WTIDX, axis = 0).copy()
                    
                    if i + (H * S) < end_H:
                        HX = np.array([windowed_X_padded[i + (j+1)*S] for j in range(H)]) # (H, T, dx)
                        
                    else: break;
                    hyper_lookback_X.append(torch.tensor(WX, dtype=torch.float32))
                    hyper_horizon_X.append(torch.tensor(HX, dtype=torch.float32))
                    hyper_lookback_ocs.append(torch.tensor(WOCS, dtype=torch.float32))
                    hyper_horizon_time.append(torch.tensor(WTIDX, dtype=torch.long))
            
            hyper_lookback_X = torch.stack(hyper_lookback_X, dim = 0)
            hyper_horizon_X = torch.stack(hyper_horizon_X,dim = 0)
            hyper_lookback_ocs = torch.stack(hyper_lookback_ocs, dim = 0)
            hyper_horizon_time = torch.stack(hyper_horizon_time,dim = 0)
            # Store results
            hyper_lookback_X_all.append(hyper_lookback_X)
            hyper_horizon_X_all.append(hyper_horizon_X)
            hyper_lookback_ocs_all.append(hyper_lookback_ocs)
            hyper_horizon_time_all.append(hyper_horizon_time)
            
            hyper_windowed_dict[key] = {
                "hyper_lookbackwindowed_X": hyper_lookback_X,  # (M, W, T, dx)
                "hyper_windowed_X": hyper_horizon_X,  # (M, H, T, dx)
                "hyper_lookbackwindowed_ocs": hyper_lookback_ocs,  # (M, W, T, dc)
                "hyper_lookbackwindowed_time": hyper_horizon_time,  # (M, W, 1)
            }
            
        hyper_lookback_X_all = torch.concat((hyper_lookback_X_all), dim = 0)
        hyper_horizon_X_all = torch.concat((hyper_horizon_X_all), dim = 0)
        hyper_lookback_ocs_all = torch.concat((hyper_lookback_ocs_all), dim = 0)
        hyper_horizon_time_all = torch.concat((hyper_horizon_time_all), dim = 0)
        return hyper_windowed_dict, (hyper_lookback_X_all, hyper_horizon_X_all, 
                                     hyper_lookback_ocs_all, hyper_horizon_time_all)

def np_to_tensor(x, ocs):
    
    data_dict = dict()
    for key in x:
        L = x[key].shape[0]
        data_dict[key] = {"X": torch.tensor(x[key], dtype=torch.float32),
                          "ocs": torch.tensor(ocs[key], dtype=torch.float32),
                          "tidx": torch.tensor(np.arange(1, L + 1).reshape(-1, 1), dtype=torch.long)}
        
    return data_dict
def format_testing_data(test_p = None, testing_ocs = None, testing_rul = None,
                        T = 25, normalize_rul = True, rectification = 125):
    full_test_set = dict()
    test_p_list = []
    test_p_ocs_list = []
    test_t_list = []
    for s in (test_p):
        Length = test_p[s].shape[0]
        if Length < T:
            # Some data of the testing machine instances are very short, so left-padding is applied 
            num_pad = T - Length
            left_padding = np.tile(test_p[s][0], (num_pad, 1)) #+ np.random.normal(0, 0.02, (num_pad, test_p[s].shape[1]))
            test_p[s] = np.vstack([left_padding, test_p[s]])
            
            left_padding = np.tile(testing_ocs[s][0], (num_pad, 1))  # + np.random.normal(0, 0.02, (num_pad, testing_ocs[s].shape[1]))
            testing_ocs[s] = np.vstack([left_padding, testing_ocs[s]])
        
        # Full test set
        windowed_CM, windowed_ocs, windowed_time = sw_function(test_p[s], testing_ocs[s], T)
        # Piece-wise linear
        rul_ = np.flip(np.arange(test_p[s].shape[0]))
        rul_ += testing_rul[s-1]
        # Rectifification
        rul_ = np.where(rul_ > rectification, rectification, rul_)
        rul_ = rul_.reshape(-1,1)
        
        if normalize_rul:
            rul_ = rul_ / rectification
        
        full_test_set[s] = {"X": torch.tensor(windowed_CM, dtype=torch.float32),
                            "Y": torch.tensor(rul_, dtype=torch.float32),
                            "ocs": torch.tensor(windowed_ocs, dtype=torch.float32),
                            "tidx": torch.tensor(windowed_time, dtype=torch.long)}
        
        # Last sample in the test set
        test_p_list.append(test_p[s][None,-T:])
        test_p_ocs_list.append(testing_ocs[s][None,-T:])
        test_t_list.append(np.array([test_p[s].shape[0]]).reshape(1,-1))
        
    test_p = torch.tensor(np.concatenate((test_p_list), axis = 0), dtype = torch.float32)
    testing_ocs = torch.tensor(np.concatenate((test_p_ocs_list), axis = 0), dtype = torch.float32)
    test_t_list = torch.tensor(np.concatenate((test_t_list), axis = 0), dtype = torch.long)
    return test_p, testing_ocs, test_t_list, full_test_set
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
    
    