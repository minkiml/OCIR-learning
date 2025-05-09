import os
import time
import numpy as np
import torch
import torch.nn as nn

from tqdm import tqdm
from pipelines import solver_base
import util_modules as ut
import src
class AEPipeline(solver_base.Solver):
    def __init__(self, config, logger):
        super(AEPipeline, self).__init__(config, logger)
        
        
        self.build_dataset()
        self.build_model()
    def __call__(self, validation = True):
        self.validation = validation
        self.train()
    
    def train(self):
        self.logger.info("======================TRAINING BEGINS======================")
        
        self.training_log = ut.Logger(self.plots_save_path, 
                            self.his_save_path,
                            ('%d', 'epoch'),
                            ('%d', 'itr'),
                            
                            ('%.5f', 'loss_rec'),
                            )
        
        Loss_R = ut.Value_averager()
        
        loss_rec = ut.Value_averager()

        opt_R = self.get_optimizers()
        vali_freq = 8  # total vali is vali_freq + 1
        vali_at = self.n_epochs // vali_freq
        self.vali("Initial")
        for epoch in tqdm(range(self.n_epochs), desc = "Training ae: "):
            self.ae.train(True)
            for i, (x,y, ocs, tidx) in enumerate(self.training_data): # TODO
                pass 
                x = x.to(self.device)
                tidx = tidx.to(self.device)
                
                if self.dc != 0:
                    if self.c_type == "discrete":
                        ocs = ut.onehot_encoding(ocs, classes= self.dc).to(self.device)
                    else:
                        ocs = ocs.to(self.device)
                else:
                    ocs = None
                
                loss_R, R = self.ae.Loss_AE(x, tidx, cond = ocs, epoch = epoch+1)
                opt_R[0].zero_grad()
                loss_R.backward()
                
                for m in reversed(opt_R):
                    if m: m.step()

                # Records
                Loss_R.update(loss_R.item())
                loss_rec.update(R[0].item())
                
                self.training_log.log_into_csv_(epoch+1,
                                            i,
                                            
                                            loss_rec.avg,
                                            )
            self.logger.info(f"epoch[{epoch+1}/{self.n_epochs}], Loss Rec:{loss_rec.avg: .4f}")
            
            if (self.validation) and (((epoch + 1) % vali_at) == 0):
                self.vali(epoch+1)
        # Save the model at the end
        self.logger.info("Training a ae is done. Saving the model ...")
        ut.save_model(self.ae, path_ = self.model_save_path, name = "AE")
        
    def vali(self, epoch): # TODO eval
        self.ae.train(False)
        
        ALL_ZE = []
        ALL_Z = []
        ALL_ZH = []
        
        ALL_CE = None
        ALL_C = []
        ALL_Q = []
        ALL_CGT = []
        
        ALL_tidx = []
        ALL_time_tokens = []
        for i, (x,_, ocs, tidx) in enumerate(self.val_data): 
            x = x.to(self.device)
            tidx = tidx.to(self.device)
            ALL_CGT.append(ocs)
            # Inference
            # z_E = self.ae.f_E.encoding(x, tidx)
            
            z_E, log_var, zin = self.ae.f_E(x, tidx)
            if self.time_embedding:
                time_tokens = self.ae.f_E.timeembedding(tidx)
                ALL_time_tokens.append(time_tokens)
            
            if self.dc != 0:
                if self.c_type == "discrete":
                    ocs = ut.onehot_encoding(ocs, classes= self.dc).to(self.device)
                else:
                    ocs = ocs.to(self.device)
            else:
                ocs = None
                        
            x_rec = self.ae.f_D(z_E, c = ocs, zin = zin)
            
            ALL_ZE.append(z_E)
            ALL_ZH = None
            ALL_Z = None

            ALL_C = None#.append(prior_c_logit if prior_c_logit is not None else prior_c)
            ALL_Q = None #.append(q_code_mu)
            
            ALL_tidx.append(tidx)
            
        # On the last batch
        self.evaluation.recon_plot(x[0,:,:], x_rec[0,:,:], label = ["true", "recon"], epoch = str(epoch))
        ALL_ZE = torch.concatenate((ALL_ZE), dim = 0)
        
        if ALL_ZH is not None:
            ALL_ZH = torch.concatenate((ALL_ZH), dim = 0)
        
        ALL_CGT = torch.concatenate((ALL_CGT), dim = 0)
        ALL_tidx = torch.concatenate((ALL_tidx), dim = 0)
        if self.time_embedding:
            ALL_time_tokens = torch.concatenate((ALL_time_tokens), dim = 0)

        self.evaluation.vae_qualitative_analysis(ALL_Z, ALL_ZH, ALL_ZE,
                                             ALL_tidx,
                                             discrete = True if self.c_type == "discrete" else False,
                                             epoch = str(epoch))
        if self.time_embedding:
            self.evaluation.vis_learned_time_embedding(ALL_time_tokens, ALL_tidx, epoch = str(epoch))
        
        # self.evaluation.
    def build_model(self):
        def print_model(m, model_name):
            self.logger.info(f"Model: {model_name}")
            total_param = 0
            for name, param in m.named_parameters():
                num_params = param.numel()
                total_param += num_params
                self.logger.info(f"{name}: {num_params} parameters")
            
            self.logger.info(f"Total parameters in {model_name}: {total_param}")
            self.logger.info("")
            return total_param
        
        ae = src.AE(dx=self.dx, dz=self.dz, dc=self.dc, window=self.window, 
                        d_model=self.d_model, num_heads=self.num_heads, z_projection=self.z_projection, 
                        D_projection=self.D_projection, time_emb=self.time_embedding, c_type=self.c_type, 
                        c_posterior_param=self.c_posterior_param, encoder_E=self.encoder_E, device=self.device)
        self.ae = ae
        print_model(ae, "AE")
        # self.ae = ut.load_model(ae, self.model_save_path, "AE")
        self.ae.to(self.device)
        
    def get_optimizers(self):
        # lr constants 
        cont_R = 0.5
        cont_Disc = 0.1
        const_G = 1.

        opt_R, scheduler_R, wd_scheduler_R = ut.opt_constructor(self.scheduler,
                                                                        [self.ae.f_E, self.ae.f_D],
                                                                       lr = self.lr_,
                                                                        warm_up = int(self.n_epochs* self.ipe * self.warm_up),
                                                                        fianl_step = int(self.n_epochs* self.ipe),
                                                                        start_lr = self.start_lr,
                                                                        ref_lr = self.ref_lr,
                                                                        final_lr = self.final_lr,
                                                                        start_wd = self.start_wd,
                                                                        final_wd = self.final_wd)
        # for opt in [ opt_R]:
        #     for i, param_group in enumerate(opt.param_groups):
        #         self.logger.info(f"Group {i}: {param_group}")
        return [opt_R, scheduler_R, wd_scheduler_R]  
    