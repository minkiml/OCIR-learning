
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
        self.time_idx = None
        
        # print(t.shape)
        # if t.dim() == 3: 
        #     self.time_idx = t.view(-1,1)
        # else:
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
                plot = None,
                valid_split = 0.
                ):
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
        
    # vis raw if necessary
    if vis and (plot is not None):
        plot.line_plot(x = training_X, label = "raw", instant = 46, inst = 1)

    # Get continous c and apply if the dataset is FD001 OR FD003
    original_X = None
    if (dataset == "FD001") or (dataset == "FD003"):
        # TODO 
        if (task == "data_trj"):
            # Keep the original CM before adding cont for comparison 
            original_X = dict()
            for key in training_X:
                original_X[key] = torch.tensor(training_X[key], dtype= torch.float32)
        phi = data_utils.get_phi(training_X)
        training_X, s_map, training_ocs, c_map = data_utils.get_continuous_property(training_X, varying = False, stds_ = None, 
                                                                                    add_cont= add_continuous, vis= True)        
        testing_X, _, testing_ocs, _ = data_utils.get_continuous_property(testing_X, varying = False, stds_ = None, 
                                                                       swapping_map = s_map, add_cont= add_continuous, vis = False)      
        if c_map is not None:
            plot.c_plot(c_map)
    # vis original scale raw + c
    if vis and (plot is not None) and ((dataset == "FD001") or (dataset == "FD003")):
        plot.line_plot(x = training_X, label = "raw_C", instant = 46, inst = 1)
    # raise NotImplementedError("")
    train_p = dict()
    train_p_ocs = dict()
    train_p_rul = dict()
    
    vali_p = dict()
    vali_p_ocs = dict()
    vali_p_rul = dict()
    
    ##################### RL and RUL estimation ########################
    if (task == "RL") or (task == "rul"):
        ''' 
        Training data is split into training (80%) and validation portion (20%) 
        and separated testing data is available for evaluation'''
        total_S = len(training_X)
    
        for s in (training_X):
            # Split machine-instance-wise
            if s <= total_S * (1. - valid_split):
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
        test_p = normalizer.transform(testing_X)
                
        training_sets = data_utils.sliding_window(train_p, train_p_ocs, train_p_rul, T) # (total num of windowed samples, w_T, c), ..., ...
        train_p, train_p_ocs, train_p_rul, train_time_idx = data_utils.parse_lr_set(training_sets)
        
        if valid_split == 0.:
            # if no validation split
            vali_p, vali_p_ocs, vali_p_rul, vali_time_idx = None, None, None, None
        else:
            vali_p = normalizer.transform(vali_p)
            vali_sets = data_utils.sliding_window(vali_p, vali_p_ocs, vali_p_rul, T)
            vali_p, vali_p_ocs, vali_p_rul, vali_time_idx = data_utils.parse_lr_set(vali_sets)
            
            
        # For rul estimation, we do not need full testing data but the last window only for rul estimation. 
        if full_test: 
            pass
        else:
            test_p, testing_ocs, test_t_list, full_test_set = data_utils.format_testing_data(test_p, testing_ocs, testing_rul,
                                                    T = T, normalize_rul=normalize_rul, rectification=rectification)
        testing_rul = torch.tensor(testing_rul, dtype = torch.float32)
        

        # max_value = max(train_time_idx.max(), vali_time_idx.max(), test_t_list.max())
        
        testing_rul = np.clip(testing_rul, 0., rectification) # isn't this cheating?
        if normalize_rul:
            train_p_rul = train_p_rul/ rectification
            if vali_p is not None:
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
        if vali_p is not None:
            logger.info(f" Validation X: {vali_p.shape}, Validation rul: {vali_p_rul.shape}, Validation ocs: {vali_p_ocs.shape}, Validation time-index: {vali_time_idx.shape}")
            vali_dataload = DataLoader(CMAPSS_datset(X = vali_p,
                                                            Y = vali_p_rul,
                                                            ocs = vali_p_ocs,
                                                            t = vali_time_idx),
                                            batch_size = batch_size,
                                            shuffle = False,
                                            num_workers = 10,
                                            drop_last=True)
        else:
            vali_dataload = None
            
        logger.info(f" Testing X: {test_p.shape}, Testing rul: {testing_rul.shape}, Testing ocs: {testing_ocs.shape}, Testing time-index: {test_t_list.shape}")
        testing_dataload = DataLoader(CMAPSS_datset(X = test_p,
                                                        Y = testing_rul,
                                                        ocs = testing_ocs,
                                                        t = test_t_list),
                                        batch_size = batch_size,
                                        shuffle = False,
                                        num_workers = 10,
                                        drop_last=False)
        return training_dataload, vali_dataload, testing_dataload, full_test_set
    
    ########################## Stationarization #############################
    elif (task == "stationarization"):
        total_S = len(training_X)
    
        for s in (training_X):
            # Split machine-instance-wise
            if s <= total_S * (1. - valid_split):
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
        test_p = normalizer.transform(testing_X)
        if valid_split == 0.:
            # if no validation split
            vali_p, vali_p_ocs, vali_p_rul, vali_time_idx = None, None, None, None
            vali_sets = None
        else:
            vali_p = normalizer.transform(vali_p)
            vali_sets = data_utils.sliding_window(vali_p, vali_p_ocs, vali_p_rul, T, in_dictionary= True)
        training_sets = data_utils.sliding_window(train_p, train_p_ocs, train_p_rul, T, in_dictionary= True)
        _, _, _, full_test_set = data_utils.format_testing_data(test_p, testing_ocs, testing_rul,
                                                    T = T, normalize_rul=normalize_rul, rectification=rectification)
        return training_sets, vali_sets, full_test_set # dictionary of all windowed sequences, ocs, etc. of shape (num samples, window, channels)
    ########################## Trajectory #############################
    elif (task == "data_trj"):
        '''
        No separated testing data is available for quantitative evaluation, so
        Training data in split into training (60%), validation (20%) and testing (20%) for quantitative eval
        Testing data (which is incomplete dataset) are used for additional qualitative eval.  
        '''
        test_p = dict()
        test_p_ocs = dict()
        test_p_rul = dict()
        
        original_train_p = dict()
        original_vali_p = dict()
        original_test_p = dict()
        
        total_S = len(training_X)
        for s in (training_X):
            # Split machine-instance-wise
            if s <= total_S * (1. - 0.3):
                train_p[s] = training_X[s]
                train_p_ocs[s] = training_ocs[s]
                train_p_rul[s] = training_rul[s]
                if original_X:
                    original_train_p[len(original_train_p) + 1] = original_X[s]   
            elif (total_S * 0.7 < s) and (s <= total_S * 0.8):
                vali_p[len(vali_p) + 1] = training_X[s]
                vali_p_ocs[len(vali_p)] = training_ocs[s]
                vali_p_rul[len(vali_p)] = training_rul[s]
                if original_X:
                    original_vali_p[len(original_vali_p) + 1] = original_X[s]        
            else:
                test_p[len(test_p) + 1] = training_X[s]
                test_p_ocs[len(test_p)] = training_ocs[s]
                test_p_rul[len(test_p)] = training_rul[s]
                if original_X:
                    original_test_p[len(original_test_p) + 1] = original_X[s]
        # Nomalization 
        normalizer = data_utils.Standardization(train_p, appr_healthy_state = 30)
        train_p = normalizer.transform(train_p)
        test_p = normalizer.transform(test_p)
        # TODO
        # normalizer_ori = data_utils.Standardization(original_train_p, appr_healthy_state = 40)
        # original_train_p = normalizer_ori.transform(original_train_p)
        # original_test_p = normalizer_ori.transform(original_test_p)   
                             
        if valid_split == 0.:
            # if no validation split
            vali_p, vali_p_ocs, vali_p_rul, vali_time_idx = None, None, None, None
            vali_sets = None
        else:
            # Validation dataset
            vali_p = normalizer.transform(vali_p)
            vali_sets = data_utils.sliding_window(vali_p, vali_p_ocs, vali_p_rul, T, in_dictionary= True)
            _, val_train_sets = data_utils.apply_hyper_window(vali_sets, H = H, W = H_lookback, S = T) # hyper window of sub window sequences
            val_train_x, val_train_y, val_train_ocs, val_train_tidx = val_train_sets # total samples
            full_val = data_utils.np_to_tensor(vali_p, vali_p_ocs)
            
            # if original_X:
            #     original_vali_p = normalizer_ori.transform(original_vali_p)
        # Training dataset
        training_sets = data_utils.sliding_window(train_p, train_p_ocs, train_p_rul, T, in_dictionary= True)
        _, training_train_sets = data_utils.apply_hyper_window(training_sets, H = H, W = H_lookback, S = T) # hyper window of sub window sequences
        trj_train_x, trj_train_y, trj_train_ocs, trj_train_tidx = training_train_sets # total samples
        
        # TODO
        # training_sets = data_utils.sliding_window(train_p, train_p_ocs, train_p_rul, T, in_dictionary= True)
        # _, training_train_sets = data_utils.apply_hyper_window(training_sets, H = H, W = H_lookback, S = T) # hyper window of sub window sequences
        # trj_train_x, trj_train_y, trj_train_ocs, trj_train_tidx = training_train_sets # total samples
        
        # Testing dataset
        testing_sets = data_utils.sliding_window(test_p, test_p_ocs, test_p_rul, T, in_dictionary= True)
        _, testing_sets = data_utils.apply_hyper_window(testing_sets, H = H, W = H_lookback, S = T) # hyper window of sub window sequences
        trj_test_x, trj_test_y, trj_test_ocs, trj_test_tidx = testing_sets # total samples
        full_test = data_utils.np_to_tensor(test_p, test_p_ocs)
        
        logger.info(f" Training X: {trj_train_x.shape}, Training rul: {trj_train_y.shape}, Training ocs: {trj_train_ocs.shape}, Training time-index: {trj_train_tidx.shape}")
        training_dataload = DataLoader(CMAPSS_datset(X = trj_train_x,
                                            Y = trj_train_y,
                                            ocs = trj_train_ocs,
                                            t = trj_train_tidx),
                            batch_size = batch_size,
                            shuffle = True,
                            num_workers = 10,
                            drop_last=True)
        if vali_p is not None:
            logger.info(f" Validation X: {val_train_x.shape}, Validation rul: {val_train_y.shape}, Validation ocs: {val_train_ocs.shape}, Validation time-index: {val_train_tidx.shape}")
            vali_dataload = DataLoader(CMAPSS_datset(X = val_train_x,
                                                            Y = val_train_y,
                                                            ocs = val_train_ocs,
                                                            t = val_train_tidx),
                                            batch_size = batch_size,
                                            shuffle = False,
                                            num_workers = 10,
                                            drop_last=True)
        else:
            vali_dataload = None
            
        logger.info(f" Testing X: {trj_test_x.shape}, Testing rul: {trj_test_y.shape}, Testing ocs: {trj_test_ocs.shape}, Testing time-index: {trj_test_tidx.shape}")
        testing_dataload = DataLoader(CMAPSS_datset(X = trj_test_x,
                                                        Y = trj_test_y,
                                                        ocs = trj_test_ocs,
                                                        t = trj_test_tidx),
                                        batch_size = batch_size,
                                        shuffle = False,
                                        num_workers = 10,
                                        drop_last=False)
        
        return training_dataload, vali_dataload, testing_dataload, (original_vali_p, original_test_p, full_val, full_test), normalizer
        