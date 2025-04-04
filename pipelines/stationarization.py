import torch
import random
import numpy as np

from tqdm import tqdm
import util_modules as ut
from datasets.cmapss import load_CMAPSS

class Stationarization(object):
    '''
    This is not a training script.
    It computes stationarization results from a trained OCIR.  
    '''
    def __init__(self, config,
                ocir, 
                logger):
        super(Stationarization, self).__init__()
        # if config['net'] == "ocir":
        self.logger = logger
        self.logger.info("Stationarization is being made ...")
        self.ocir = ocir
        self.dc = config['dc']
        dev = config['gpu_dev']
        self.device = torch.device(f'cuda:{dev}' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"GPU (device: {dev}) used" if torch.cuda.is_available() else 'cpu used')
        self.evaluation = ut.Evaluation(config['plots_save_path'],
                                        config['his_save_path'],
                                        self.logger)
                    
        self.build_dataset(config)
        self.stationarization_test()
        # else:
        #     pass
    def stationarization_test(self):
        num_st = 5
        self.ocir.train(False)

        N = len(self.training_data)
        inst = num_st
        if inst > N:
            raise ValueError("")
        indx = random.sample(range(1, N + 1), inst)
        with torch.no_grad():
            for ii in tqdm(indx, desc = "Stationarization ... "):
                x = self.training_data[ii]["X"].to(self.device)
                tidx = self.training_data[ii]["t_idx"].to(self.device)
                stationarized_X, stationarized_X_from_G = self.ocir.stationarization(x, tidx)
                x = x[:,-1,:] 
                
                stationarized_X = stationarized_X[:,-1,:] # un-windowing
                self.evaluation.recon_plot(x, stationarized_X, label = ["Observed",  r"Stationarized"], 
                                           epoch = str(ii), title = "st_Dec", plot_s = True)
                if stationarized_X_from_G is not None:
                    stationarized_X_from_G = stationarized_X_from_G[:,-1,:]
                    self.evaluation.recon_plot(x, stationarized_X_from_G, label = ["Observed",  r"Stationarized"], 
                                               epoch = str(ii), title = "st_Gen", plot_s = True)
    def build_dataset(self, config):
        # Dataset instantiation
        if config['dataset'] == "circle": 
            raise NotImplementedError("")
            load_circle()
        else: 
            self.training_data, self.val_data, self.testing_data = load_CMAPSS(dataset = config['dataset'],
                                                                    data_path = config['data_path'],
                                                                    task = "stationarization",
                                                                    T = config['window'],
                                                                    H = config['H'],
                                                                    H_lookback = config['hyper_lookback'],
                                                                    rectification = 125,
                                                                    batch_size = config['batch'],
                                                                    normalize_rul = True,
                                                                    vis = False,
                                                                    logger = self.logger,
                                                                    plot = self.evaluation)