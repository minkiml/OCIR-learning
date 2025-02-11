import os
import time
import numpy as np
import torch
import torch.nn as nn

from tqdm import tqdm
from pipelines import solver_base
from util_modules.logger import Value_averager, Logger, grad_logger_spec

class RL_pipeline(solver_base.Solver):
    def __init__(self, config, logger):
        super(RL_pipeline, self).__init__(config, logger)
        
        
        self.build_dataset()
        self.build_model()
        
        #Early stopping 
        self.counter = 0
        self.metric_1 = 1 # TODO
        self.metric_2 = 2 # TODO 
    def __call__(self):
        pass
    
    def train(self):
        self.logger.info("======================TRAINING BEGINS======================")
        
        self.training_log = Logger(self.plots_save_path, 
                            self.his_save_path,
                            ('%d', 'epoch'),
                            ('%d', 'itr'),
                            ('%.5f', 'loss_r'),
                            ('%.5f', 'loss_g'),
                            ('%.5f', 'loss_r_g'),
                            ('%.5f', 'loss_total'),
                            
                            ('%.5f', 'lr'),
                            ('%.5f', 'wd'),
                            ('%.4e', 'encoder'), 
                            )
        
        loss_r = Value_averager()
        loss_g = Value_averager()
        loss_r_g = Value_averager()
        loss_total = Value_averager()
        
        criteria = 1 # TODO
        self.get_optimizers() # TODO
        for epoch in tqdm(range(self.n_epochs), desc = "Training OCIR: "):
            self.ocir.train()
            for i, (x,_, _, tidx) in enumerate(self.training_data): # TODO
                pass 
                x.to(self.device)
                tidx.to(self.device)
                loss_L_R = self.ocir.L_R(x, tidx = tidx)
                #backward and step 
                loss_L_G1 = self.ocir.L_G_discriminator(x)
                # backward and step
                loss_L_G2 = self.ocir.L_G_generator(x.shape[0])
                # backward and step
                

            #     loss_r.update(TD_loss.item())
            #     loss_g.update(FD_loss.item())
            #     loss_r_g.update(loss.item())
            #     loss_total.update(loss.item())
            # self.logger.info(f"epoch[{epoch+1}/{self.n_epochs}, 
            #                  Loss_TD:{self.loss_TD.avg: .4f} , 
            #                  Loss_FD:{self.loss_FD.avg: .4f}, 
            #                  Loss_total: {self.loss_total.avg: .4f}")
            
            #TODO: validation 
            #TODO: early stopping check and saving check point
        # Save the model at the end
    def eval(self):
        pass
    
    def early_stop(self, measure_1, measure_2):
        pass
        # TODO
        if 1:
            # Save the model check point here TODO
            pass
    
    def get_optimizers(self):
        # TODO
        return super().get_optimizers() 
    