import os
import time
import numpy as np
import torch
import torch.nn as nn

from tqdm import tqdm
from pipelines import solver_base
from util_modules import logger, optimizer, utils

class RL_pipeline(solver_base.Solver):
    def __init__(self, config, logger):
        super(RL_pipeline, self).__init__(config, logger)
        
        
        self.build_dataset()
        self.build_model()
        
    def __call__(self, validation = False):
        self.validation = validation
        self.train()
    
    def train(self):
        self.logger.info("======================TRAINING BEGINS======================")
        
        self.training_log = logger.Logger(self.plots_save_path, 
                            self.his_save_path,
                            ('%d', 'epoch'),
                            ('%d', 'itr'),
                            ('%.5f', 'loss_rec'),
                            ('%.5f', 'loss_kl'),
                            
                            ('%.5f', 'loss_realD'),
                            ('%.5f', 'loss_fakeD'),
                            
                            ('%.5f', 'loss_gen'),
                            ('%.5f', 'loss_Q'),
                            ('%.5f', 'loss_ccz'),
                            ('%.5f', 'loss_ccc'),
                            )
        
        Loss_R = logger.Value_averager()
        Loss_Disc = logger.Value_averager()
        Loss_G = logger.Value_averager()
        
        # In order to track the losses 
        loss_rec = logger.Value_averager()
        loss_kl = logger.Value_averager()
        
        loss_realD = logger.Value_averager()
        loss_fakeD = logger.Value_averager()
        
        loss_gen = logger.Value_averager()
        loss_Q = logger.Value_averager()
        loss_ccz = logger.Value_averager()
        loss_ccc = logger.Value_averager()
                
        opt_R, opt_Disc, opt_G = self.get_optimizers()
        for epoch in tqdm(range(self.n_epochs), desc = "Training OCIR: "):
            self.ocir.train(True)
            for i, (x,_, _, tidx) in enumerate(self.training_data): # TODO
                pass 
                x.to(self.device)
                tidx.to(self.device)
                
                # (1)
                loss_R, R = self.ocir.L_R(x, tidx)
                opt_R[0].zero_grad()
                loss_R.backward()
                for m in reversed(opt_R):
                    if m: m.step()
                                
                # (2)
                loss_disc, D = self.ocir.L_G_discriminator(x)
                opt_Disc[0].zero_grad()
                loss_disc.backward()
                for m in reversed(opt_Disc):
                    if m: m.step()
                
                # (3)
                loss_G, G = self.ocir.L_G_generator(x)
                opt_G[0].zero_grad()
                loss_G.backward()
                for m in reversed(opt_Disc):
                    if m: m.step()
                

                # Records
                Loss_R.update(loss_R.item())
                Loss_Disc.update(loss_disc.item())
                Loss_G.update(loss_G.item())
                
                loss_rec.update(R[0].item())
                loss_kl.update(R[1].item())
                
                loss_realD.update(D[0].item())
                loss_fakeD.update(D[1].item())
                
                loss_gen.update(G[0].item())
                loss_Q.update(G[1].item())
                loss_ccz.update(G[2].item())
                loss_ccc.update(G[3].item())
                self.training_log.log_into_csv_(epoch+1,
                                            i,
                                            self.loss_rec.avg,
                                            self.loss_kl.avg,
                                            
                                            self.loss_realD.avg,
                                            self.loss_fakeD.avg,
                                            
                                            self.loss_gen.avg,
                                            self.loss_Q.avg,
                                            self.loss_ccz.avg,
                                            self.loss_ccc.avg,
                                            )

            self.logger.info(f"epoch[{epoch+1}/{self.n_epochs}, 
                             Loss R:{self.Loss_R.avg: .4f} , 
                             Loss Disc:{self.Loss_Disc.avg: .4f}, 
                             Loss G: {self.Loss_G.avg: .4f}")
            
            if self.validation :
                self.vali()
            
        # Save the model at the end
        self.logger.info("Training a OCIR is done. Saving the model ...")
        utils.save_model(self.ocir, path_ = self.model_save_path, name = "OCIR")
        
    def vali(self):
        # Note that it is "not necessary" as we are doing unsupervised learning. 
        # We mainly check whether the objective functions converge or not.
        Loss_R_vali = logger.Value_averager()
        Loss_Disc_vali = logger.Value_averager()
        Loss_G_vali = logger.Value_averager()
        self.ocir.train(False)
        for i, (x,_, _, tidx) in enumerate(self.val_data): 
            x.to(self.device)
            tidx.to(self.device)
            
            loss_R, _ = self.ocir.L_R(x, tidx)            
            loss_disc, _ = self.ocir.L_G_discriminator(x)
            loss_G, _ = self.ocir.L_G_generator(x)
            
            Loss_R_vali.update(loss_R.item())
            Loss_Disc_vali.update(loss_disc.item())
            Loss_G_vali.update(loss_G.item())
        
        self.logger.info(f"Vali: 
                        Loss R:{self.Loss_R_vali.avg: .4f} , 
                        Loss Disc:{self.Loss_Disc_vali.avg: .4f}, 
                        Loss G: {self.Loss_G_vali.avg: .4f}")

    def get_optimizers(self):
        opt_R, scheduler_R, wd_scheduler_R = optimizer.opt_constructor([self.ocir.f_E, self.ocir.f_C, self.ocir.f_D, self.ocir.h])
        opt_Disc, scheduler_Disc, wd_scheduler_Disc = optimizer.opt_constructor([self.ocir.D])
        opt_G, scheduler_G, wd_scheduler_G = optimizer.opt_constructor([self.ocir.f_E, self.ocir.f_C, self.ocir.G, self.ocir.h])

        return [opt_R, scheduler_R, wd_scheduler_R], \
               [opt_Disc, scheduler_Disc, wd_scheduler_Disc], \
               [opt_G, scheduler_G, wd_scheduler_G]     
    