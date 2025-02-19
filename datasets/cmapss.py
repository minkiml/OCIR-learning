
import os
import pickle
import torch
import numpy as np

from datasets.data_format import data_utils, cmapssformater
from torch.utils.data import Dataset, DataLoader

class CMAPSS_datset(Dataset):
    '''
    X (input to model): (num_samples, window, channel)
    Y (rul): (num_samples, 1)
    ocs: (num_samples, window, ocs_dim)
    t (time index): (num_samples, 1)
    '''
    def __init__(self, 
            X,
            Y,
            ocs,
            t = None):
        super(CMAPSS_datset, self).__init__()

        self.X = X #[:,:,0:1] # (n, w_T, channel)
        self.Y = Y # (n, 1) # TODO check shape 
        self.ocs = ocs # (n, w_T, c)
        self.time_idx = t.view(-1,1) # (N, 1)
    def __len__(self):
        return self.X.shape[0] 
    
    def __getitem__(self, index):
        if self.time_idx is not None:
            return self.X[index], self.Y[index], self.ocs[index], self.time_idx[index], 
        else:
            return self.X[index], self.Y[index], self.ocs[index]

class CMAPSS_trajectory_datset(Dataset):
    def __init__(self, 
            X,
            Y,
            ocs_X,
            ocs_Y,
            t = None):
        super(CMAPSS_trajectory_datset, self).__init__()

        self.X = X # (n, w_T, channel)
        self.Y = Y # (n, wT_H, channel)  
        self.ocs_X = ocs_X # (n, w_T, c)
        self.ocs_Y = ocs_Y 
        self.time_idx = t # (n, 1, 1)
    def __len__(self):
        return self.X.shape[0] 
    
    def __getitem__(self, index):
        return self.X[index], self.Y[index], self.ocs_X[index], self.ocs_Y[index], self.time_idx[index], 

        
def load_CMAPSS(dataset = "FD001",
                data_path = "",
                task = "",
                
                T = 25,
                H = 2,
                H_lookback = 2,
                rectification = 125,
                batch_size = 32,
                normalize_rul = True,
                full_test = False,
                vis = False,
                logger = None,
                plot = None):
    add_continuous = True
    # Load the raw CM data 
    path = os.path.join(data_path, dataset)
    if not os.path.exists(os.path.join(path, 'training_data.pkl')):
        # Get the formatted raw CM data. Download the raw CM data from (... TODO)
        logger.info("Formatted CMAPSS data does not exist, so being formatted first ...")
        cmapssformater.format_CMAPSS(rectification = rectification) # the path for the downloaded raw cm data need to be speificed inside the function 
    
    with open(os.path.join(path, 'training_data.pkl'), 'rb') as f:
        training_X = pickle.load(f)
        
    if os.path.exists(os.path.join(path, 'training_ocs.pkl')):
        with open(os.path.join(path, 'training_ocs.pkl'), 'rb') as f:
            training_ocs = pickle.load(f)
    
    with open(os.path.join(path, 'training_rul.pkl'), 'rb') as f:
        training_rul = pickle.load(f)
        
    with open(os.path.join(path, 'testing_data.pkl'), 'rb') as f:
        testing_X = pickle.load(f)
    
    if os.path.exists(os.path.join(path, 'testing_ocs.pkl')):
        with open(os.path.join(path, 'testing_ocs.pkl'), 'rb') as f:
            testing_ocs = pickle.load(f)

    with open(os.path.join(path, 'testing_rul.pkl'), 'rb') as f:
        testing_rul = pickle.load(f)
        
    # vis RAW here if necessary
    if vis and (plot is not None):
        plot.line_plot(x = training_X, label = "raw", instant = 0, inst = 6)

    # Get continous c and apply if the dataset is FD001 OR FD003
    if (dataset == "FD001") or (dataset == "FD003"):
        phi = data_utils.get_phi(training_X)
        training_X, s_map, training_ocs= data_utils.get_continuous_property(training_X, varying = False, stds_ = None, add_cont= add_continuous)        
        testing_X, _, testing_ocs = data_utils.get_continuous_property(testing_X, varying = False, stds_ = None, swapping_map = s_map, add_cont= add_continuous)      

    # VIS original scale raw + c data
    if vis and (plot is not None) and ((dataset == "FD001") or (dataset == "FD003")):
        plot.line_plot(x = training_X, label = "raw_C", instant = 0, inst = 6)

    train_p = dict()
    train_p_ocs = dict()
    train_p_rul = dict()
    
    vali_p = dict()
    vali_p_ocs = dict()
    vali_p_rul = dict()
        
    if (task == "RL") or (task == "rul_estimation"):
        ''' 
        Training data is split into training (80%) and validation portion (20%) 
        and separated testing data is available for evaluation'''
        total_S = len(training_X)
    
        for s in (training_X):
            # Split machine-instance-wise
            if s <= total_S * 0.8:
                train_p[s] = training_X[s]
                train_p_ocs[s] = training_ocs[s]
                train_p_rul[s] = training_rul[s]
                
            else:
                vali_p[len(vali_p) + 1] = training_X[s]
                vali_p_ocs[len(vali_p)] = training_ocs[s]
                vali_p_rul[len(vali_p)] = training_rul[s]
                
        # Nomalization 
        normalizer = data_utils.Standardization(train_p, appr_healthy_state = 30)
        train_p = normalizer.transform(train_p)
        vali_p = normalizer.transform(vali_p)
        test_p = normalizer.transform(testing_X)
        
        # Windowing TODO check rul shpae  # SHAPE CHECKING
        training_sets, _= data_utils.sliding_window(train_p, train_p_ocs, train_p_rul, T) # (total num of windowed samples, w_T, c), ..., ... # TODO check rul in sw
        vali_sets, _ = data_utils.sliding_window(vali_p, vali_p_ocs, vali_p_rul, T)
        train_p, train_p_ocs, train_p_rul, train_time_idx = data_utils.parse_lr_set(training_sets)
        vali_p, vali_p_ocs, vali_p_rul, vali_time_idx = data_utils.parse_lr_set(vali_sets)
        # TODO check the size the vali
        # Note that for rul estimation, we do not need full testing data but the last window only. 
        if full_test:
            raise NotImplementedError("")
        else:
            test_p_list = []
            test_p_ocs_list = []
            test_t_list = []
            for s in (test_p):
                if test_p[s].shape[0] >= T:
                    pass
                else:
                    # Some data of the testing machine instances are very short, so left-padding is applied 
                    num_pad = T - test_p[s].shape[0]
                    left_padding = np.tile(test_p[s][0], (num_pad, 1)) + np.random.normal(0, 0.02, (num_pad, test_p[s].shape[1]))
                    test_p[s] = np.vstack([left_padding, test_p[s]])
                    
                    left_padding = np.tile(testing_ocs[s][0], (num_pad, 1)) + np.random.normal(0, 0.02, (num_pad, testing_ocs[s].shape[1]))
                    testing_ocs[s] = np.vstack([left_padding, testing_ocs[s]])
                    
                test_p_list.append(test_p[s][None,-T:])
                test_p_ocs_list.append(testing_ocs[s][None,-T:])
                test_t_list.append(np.array([test_p[s].shape[0]]).reshape(1,-1))
            test_p = torch.tensor(np.concatenate((test_p_list), axis = 0), dtype = torch.float32)
            testing_ocs = torch.tensor(np.concatenate((test_p_ocs_list), axis = 0), dtype = torch.float32)
            test_t_list = torch.tensor(np.concatenate((test_t_list), axis = 0), dtype = torch.long)
        testing_rul = torch.tensor(testing_rul, dtype = torch.float32)
        

        # max_value = max(train_time_idx.max(), vali_time_idx.max(), test_t_list.max())
        
        #rul_test = np.clip(rul_test, 0., rectification) # isn't this cheating?
        if normalize_rul:
            train_p_rul = train_p_rul/ rectification
            vali_p_rul = vali_p_rul/ rectification
            testing_rul = testing_rul / rectification
        logger.info(f" Training X: {train_p.shape}, Training rul: {train_p_rul.shape}, Training ocs: {train_p_ocs.shape}, Training time-index: {train_time_idx.shape}")
        training_dataload = DataLoader(CMAPSS_datset(X = train_p,
                                            Y = train_p_rul,
                                            ocs = train_p_ocs,
                                            t = train_time_idx),
                            batch_size = batch_size,
                            shuffle = True,
                            num_workers = 10,
                            drop_last=True)
        logger.info(f" Validation X: {vali_p.shape}, Validation rul: {vali_p_rul.shape}, Validation ocs: {vali_p_ocs.shape}, Validation time-index: {vali_time_idx.shape}")
        vali_dataload = DataLoader(CMAPSS_datset(X = vali_p,
                                                        Y = vali_p_rul,
                                                        ocs = vali_p_ocs,
                                                        t = vali_time_idx),
                                        batch_size = batch_size,
                                        shuffle = False,
                                        num_workers = 10,
                                        drop_last=True)
        logger.info(f" Testing X: {test_p.shape}, Testing rul: {testing_rul.shape}, Testing ocs: {testing_ocs.shape}, Testing time-index: {test_t_list.shape}")
        testing_dataload = DataLoader(CMAPSS_datset(X = test_p,
                                                        Y = testing_rul,
                                                        ocs = testing_ocs,
                                                        t = test_t_list),
                                        batch_size = batch_size,
                                        shuffle = False,
                                        num_workers = 10,
                                        drop_last=False)
        return training_dataload, vali_dataload, testing_dataload
    elif (task == "rep_trj") or (task == "data_trj"):
        '''
        No separated testing data is available for quantitative evaluation, so
        Training data in split into training (60%), validation (20%) and testing (20%) for quantitative eval
        Testing data (which is incomplete dataset) are used for additional qualitative eval.  
        '''
        test_p = dict()
        test_p_ocs = dict()
        test_p_rul = dict()
        
        total_S = len(training_X)
    
        for s in (training_X):
            # Split machine-instance-wise
            if s <= total_S * 0.6:
                train_p[s] = training_X[s]
                train_p_ocs[s] = training_ocs[s]
                train_p_rul[s] = training_rul[s]
                
            elif (total_S * 0.6 < s) and (s <= total_S * 0.8):
                vali_p[len(vali_p) + 1] = training_X[s]
                vali_p_ocs[len(vali_p)] = training_ocs[s]
                vali_p_rul[len(vali_p)] = training_rul[s]
            
            else:
                test_p[len(test_p) + 1] = training_X[s]
                test_p_ocs[len(test_p)] = training_ocs[s]
                test_p_rul[len(test_p)] = training_rul[s]
                
        # Nomalization 
        normalizer = data_utils.Standardization(train_p, appr_healthy_state = 30)
        train_p = normalizer.transform(train_p)
        vali_p = normalizer.transform(vali_p)
        test_p = normalizer.transform(testing_X)

        # Sliding window (TODO)
        train_lr_set, train_trj_set = data_utils.sliding_window(train_p, train_p_ocs, train_p_rul, T,
                                                                                    H = H, H_lookback = H_lookback, 
                                                                                    trajectory = "feature" if task == "rep_trj" else "data") # (total num of windowed samples, w_T, c), ..., ... # TODO check rul in sw
        
        train_p, train_p_ocs, train_p_rul, train_time_idx = data_utils.parse_lr_set(train_lr_set)
        (train_lookback_p, train_lookback_p_ocs, train_lookback_p_tidx), (train_horizon_p, train_horizon_p_ocs) = data_utils.parse_trj_set(train_trj_set)
        
        vali_lr_set, vali_trj_set = data_utils.sliding_window(vali_p, vali_p_ocs, vali_p_rul, T,
                                                                                H = H, H_lookback = H_lookback, 
                                                                                trajectory = "feature" if task == "rep_trj" else "data") # (total num of windowed samples, w_T, c), ..., ...
        vali_p, vali_p_ocs, vali_p_rul, vali_time_idx = data_utils.parse_lr_set(vali_lr_set)
        (vali_lookback_p, vali_lookback_p_ocs, vali_lookback_p_tidx), (vali_horizon_p, vali_horizon_p_ocs) = data_utils.parse_trj_set(vali_trj_set)

        test_lr_set, test_trj_set = data_utils.sliding_window(test_p, test_p_ocs, test_p_rul, T,
                                                                                    H = H, H_lookback = H_lookback, 
                                                                                    trajectory = "feature" if task == "rep_trj" else "data") # (total num of windowed samples, w_T, c), ..., ...
        test_p, test_p_ocs, test_p_rul, test_time_idx = data_utils.parse_lr_set(test_lr_set)
        (test_lookback_p, test_lookback_p_ocs, test_lookback_p_tidx), (test_horizon_p, test_horizon_p_ocs) = data_utils.parse_trj_set(test_trj_set)
        # test_p, test_p_ocs, test_p_rul, test_time_idx

        # TODO format the above into dataloader
        raise NotImplementedError("todo")
        return training_dataload, vali_dataload, testing_dataload