import pandas as pd
import numpy as np
import pickle
import os

from tqdm import tqdm
from sklearn.cluster import KMeans
from datasets.data_format import data_utils
from util_modules import utils

def format_CMAPSS(rectification = 125):
    '''
    Loads CAPSS datasets (FD001, ..., FD004) and format as training and testing data
    - Used 14 channels 
    - 3 oerpational parameters are available and used for discrete case 
             
    This yields: 
    
    1) a set of training datasets in 3 dictionaries
    - S number of (L by 14) full run-to-failure sequences
    - S number of (L by C) operationg condition labels (which is used for evaluation only) in case of fd002 or fd004 (discrete case)
    - S number of (L by 1) rul labels 
    
    2) a set of testing datasets in 3 dictionaries
    - N number of (L by 14) incomplete sequence
    - N number of (L by C) operation condition labels (which is used for evaluation only) in case of fd002 or fd004 (discrete case)
    - N number of (1, ) true rul label measured at the last oberservation x_t in each machine  -> Note that this is in numpy array
    
    Save all the formated data under data_format folder
    '''
    
    path_ = '/data/home/mkim332/data/CMAPSS'  # Give the path for raw unformatted txt cmapss data  #TODO hide this when pushing 
    data_dir = "./datasets/cmapss_dataset"
    
    
    '''Training data'''
    datasets = ["FD001", "FD002", "FD003", "FD004"]
    for j, dataset in tqdm(enumerate(datasets), desc = "Formating CMAPSS data.."):
        
        training_data = dict()
        ocs_training = dict() 
        rul_training = dict()
        
        testing_data = dict()
        ocs_testing = dict() # used for baselines
        
        save_to_path = os.path.join(data_dir, dataset)
        utils.mkdir(save_to_path)
        
        file_name = "train_" + dataset
        file_ = path_ + file_name + ".txt"
        df = pd.read_csv(f'{file_}',delimiter = ' ', header=None,
                                usecols = [0, 2, 3, 4, 
                                           6, 7, 8, 11, 
                                            12, 13, 15, 16, 17, 18, 
                                            19, 21, 24, 25]) # 14 channels + machine entity id (first column) + true operating conditions (2, 3, 4)
        
        # Get unique entity labels and their counts
        entity_labels = df[0].values
        entity_ids, L = np.unique(entity_labels, return_counts=True)

        # Numpify
        np_array = df.iloc[:,4:].to_numpy()
        
        # only get ocs for testing data for eval purpose
        if (dataset == "FD002") or (dataset == "FD004"):
            np_ocs = df.iloc[:,1:4].to_numpy()
            # K-mean to reformat the operational settings into class labels
            kmeans = KMeans(6)
            np_ocs = kmeans.fit_predict(np_ocs).reshape(1,-1)
        else: 
            np_ocs = None
            ocs_training = None
        # else: 
        #     '''
        #     For FD001 and FD003 (constant operating condition), we add φc_t to have continous operating regimes
        #     '''
        #     phi = data_utils.get_phi(np_array, entity_ids, entity_labels)
        #     cont_prop, s_map, np_ocs= data_utils.get_continuous_property(np_array.shape, varying = True, stds_ = phi)        
        #     original_np = np.array(np_array) # original sequence
        #     np_array = np_array + cont_prop # x + φc_t
        #     np_ocs = np_ocs.reshape(-1,1) # ground truth ocs
        #     # plot_data(np_array[np.where(entity_labels == 65)[0]], title_ = "cont")

    
        for i, label in enumerate(entity_ids):
            # Training data 100 % 
            indices = np.where(entity_labels == label)[0]
            training_data[i+1] = np_array[indices, :]
            if np_ocs is not None:
                ocs_training[i+1] = np_ocs[indices]

            # Piece-wise linear
            rul_ = np.flip(np.arange(L[i]))
            # Rectifification
            rul_ = np.where(rul_ > rectification, rectification, rul_)
            rul_training[i+1] = rul_.reshape(-1,1)
        
        '''Testing data with true RUL labels ''' 
        # Seperate incomplete testing data
        file_name = "test_"+dataset
        file_ = path_ + file_name + ".txt"
        df = pd.read_csv(f'{file_}',delimiter = ' ', header=None,
                                usecols = [0, 2, 3, 4, 
                                            6, 7, 8, 11, # 7, 8, 11, 15, 16 
                                            12, 13, 15, 16, 17, 18, 
                                            19, 21, 24, 25]) # 14 channels + machine entity id (first column)
        entity_labels = df[0].values
        # Get unique entity labels and their counts
        entity_ids, L = np.unique(entity_labels, return_counts=True)
        np_array = df.iloc[:,4:].to_numpy()

            # FD002 and FD004
        if (dataset == "FD002") or (dataset == "FD004"):
            np_ocs = df.iloc[:,1:4].to_numpy()
            # K-mean to reformat the operational settings into class labels
            np_ocs = kmeans.fit_predict(np_ocs).reshape(1,-1)
        else: # FD001 and FD003
            np_ocs = None
            ocs_testing = None
            # cont_prop, _, np_ocs = data_utils.get_continuous_property(np_array.shape, varying = True, stds_ = phi, swapping_map = s_map)        
            # test_original_np = np.array(np_array) # original sequence
            # np_array = np_array + cont_prop # x + φc_t
            # np_ocs = np_ocs.reshape(-1,1)
            
        for i, label in enumerate(entity_ids):
            indices = np.where(entity_labels == label)[0]
            testing_data[i+1] = np_array[indices, :]
            if np_ocs is not None:
                ocs_testing[i+1] = np_ocs[indices]

        # Get true RUL labels (no need for piece-wise linear)
        file_name = "RUL_"+dataset
        file_ = path_ + file_name + ".txt"
        testing_rul = pd.read_csv(f'{file_}',delimiter = ' ', header=None, usecols=[0]).to_numpy()
        
        # Save the data ## 
        os.path.join(save_to_path, "training_data.pkl")
        with open(os.path.join(save_to_path, "training_data.pkl"), 'wb') as f:
            pickle.dump(training_data, f)
        
        if ocs_training:
            with open(os.path.join(save_to_path, "training_ocs.pkl"), 'wb') as f:
                pickle.dump(ocs_training, f)
        
        with open(os.path.join(save_to_path, "training_rul.pkl"), 'wb') as f:
            pickle.dump(rul_training, f)


        with open(os.path.join(save_to_path, "testing_data.pkl"), 'wb') as f:
            pickle.dump(testing_data, f)
            
        if ocs_testing:
            with open(os.path.join(save_to_path, "testing_ocs.pkl"), 'wb') as f:
                pickle.dump(ocs_testing, f)
            
        with open(os.path.join(save_to_path, "testing_rul.pkl"), 'wb') as f:
            pickle.dump(testing_rul, f)
        