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
        if config['net'] == "ocir":
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
        else:
            pass
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
                if self.ocir.shared_encoder_layers is not None:
                    h = self.ocir.shared_encoder_layers(x, tidx)
                    hc = h
                    if (self.ocir.z_projection == "spc") or (self.ocir.z_projection == "seq"):
                        hc = hc[:,1:,:]
                    if self.ocir.time_emb:
                        hc = hc[:,:-1,:]
                else: 
                    h = x
                    hc = x
                
                mu, log_var, _ = self.ocir.f_E(h)
                z, _, _ = self.ocir.h(z0 = mu)
                
                # Fixed code
                N_c = x.shape[0:-1] + (self.dc,)

                fixed_c = self.ocir.prior_c.sample(N_c, target = 0.1 if self.ocir.c_type == "continuous" else 1)
                
                # Reconstruction
                stationarized_X = self.ocir.f_D(z, c = fixed_c, zin = None, generation = True)
                stationarized_X = stationarized_X[:,-1,:] # un-windowing
                stationarized_X_from_G = self.ocir.G(z, c = fixed_c, zin = None, generation = True)
                stationarized_X_from_G = stationarized_X_from_G[:,-1,:]
                
                x = x[:,-1,:]
                self.evaluation.recon_plot(x, stationarized_X, label = ["true", "stationarization - f_D"], epoch = str(ii), title = "st_Dec")
                self.evaluation.recon_plot(x, stationarized_X_from_G, label = ["true", "stationarization - G"], epoch = str(ii), title = "st_Gen")
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