import os
import time
import numpy as np
import torch
import torch.nn as nn

from tqdm import tqdm
from pipelines import solver_base
import util_modules as ut
import src
class VAEPipeline(solver_base.Solver):
    # supervised = True
    def __init__(self, config, logger):
        super(VAEPipeline, self).__init__(config, logger)
        
        self.supervised = bool(self.conditional)
        
        self.build_dataset()
        self.build_model()
    def __call__(self, validation = True):
        self.validation = validation
        if self.required_training:
            self.train()
        else:
            self.vali("validation")
        self.vae.train(False)
        return self.vae
    
    def train(self):
        self.logger.info("======================TRAINING BEGINS======================")
        
        self.training_log = ut.Logger(self.plots_save_path, 
                            self.his_save_path,
                            ('%d', 'epoch'),
                            ('%d', 'itr'),
                            
                            ('%.5f', 'loss_rec'),
                            ('%.5f', 'loss_kl')
                            )
        
        Loss_R = ut.Value_averager()
        
        loss_rec = ut.Value_averager()
        loss_kl = ut.Value_averager()

        opt_R = self.get_optimizers()
        vali_freq = 8  # total vali is vali_freq + 1
        vali_at = self.n_epochs // vali_freq
        self.vali("Initial")
        for epoch in tqdm(range(self.n_epochs), desc = "Training VAE: "):
            self.vae.train(True)
            for i, (x,y, ocs, tidx) in enumerate(self.training_data): # TODO
                pass 
                x = x.to(self.device)
                tidx = tidx.to(self.device)
                if self.dc != 0 or self.supervised:
                    if self.supervised:
                        if self.c_type == "discrete":
                            ocs = ut.onehot_encoding(ocs, classes= self.dc).to(self.device)
                        else:
                            ocs = ocs.to(self.device)
                    else:
                        ocs = None#self.vae.f_C.inference(x) # TODO
                else:
                    ocs = None
                        
                if self.vae.h is None:
                    loss_R, R = self.vae.Loss_VAE(x, tidx, cond = ocs, epoch = epoch+1)
                else:
                    loss_R, R = self.vae.L_R(x, tidx, cond = ocs, epoch = epoch+1)
                opt_R[0].zero_grad()
                loss_R.backward()
                
                # for name, param in self.vae.h.time_embedding.named_parameters():
                #     if param.grad is not None:
                #         print(f"Epoch {epoch}, {name} grad norm: {param.grad.norm().item()} \n")
                #     else:
                #         print("None \n")
                for m in reversed(opt_R):
                    if m: m.step()

                # Records
                Loss_R.update(loss_R.item())
                loss_rec.update(R[0].item())
                loss_kl.update(R[1].item())
                
                self.training_log.log_into_csv_(epoch+1,
                                            i,
                                            
                                            loss_rec.avg,
                                            loss_kl.avg,
                                            )
            self.logger.info(f"epoch[{epoch+1}/{self.n_epochs}], Loss Rec:{loss_rec.avg: .4f}, Loss KL:{loss_kl.avg: .4f}")
            if vali_at > 0:
                if (self.validation) and (((epoch + 1) % vali_at) == 0):
                    self.vali(epoch+1)
        # Save the model at the end
        self.logger.info("Training a VAE is done. Saving the model ...")
        ut.save_model(self.vae, path_ = self.model_save_path, name = "vae")
        
    def vali(self, epoch): # TODO eval
        self.vae.train(False)
        
        ALL_ZE = []
        ALL_Z = []
        ALL_ZH = []
        
        ALL_ZH_Z0 = []
        ALL_C = []
        ALL_Q = []
        ALL_CGT = []
        
        ALL_tidx = []
        ALL_time_tokens = []
        with torch.no_grad():
            for i, (x,_, ocs, tidx) in enumerate(self.training_data): 
                x = x.to(self.device)
                tidx = tidx.to(self.device)
                ALL_CGT.append(ocs)
                if self.dc != 0 or self.supervised:
                    if self.supervised:
                        if self.c_type == "discrete":
                            ocs = ut.onehot_encoding(ocs, classes= self.dc).to(self.device)
                        else:
                            ocs = ocs.to(self.device)
                    else:
                        ocs = None#self.vae.f_C.inference(x)
                else:
                    ocs = None
                # Inference
                # z_E = self.vae.f_E.encoding(x, tidx)
                if self.vae.h is None:
                    z_E, log_var, zin = self.vae.f_E(x, tidx)
                    z_E_on_z,_, _ = self.vae.f_E.reparameterization_NF(z_E, log_var)
                    
                    if self.time_embedding:
                        time_tokens = self.vae.f_E.timeembedding(tidx)
                        ALL_time_tokens.append(time_tokens)
                    
                    x_rec = self.vae.f_D(z_E, c = ocs, zin = zin)
                    ALL_ZE.append(z_E)
                    ALL_ZH.append(z_E_on_z)
                    ALL_Z = None
                    # ALL_ZH.append(z_h)
                
                    # ALL_ZH.append(z_h)
                    # ALL_Z.append(prior_z)
                    ALL_C = None#.append(prior_c_logit if prior_c_logit is not None else prior_c)
                    ALL_Q = None #.append(q_code_mu)
                    # ALL_CGT.append(ocs)
                    
                    ALL_tidx.append(tidx)
                else:
                    z_E, log_var, zin = self.vae.f_E(x, tidx)
                    z_H_z0,_, z0 = self.vae.f_E.reparameterization_NF(z_E, log_var)
                    
                    z_H, _, _ = self.vae.f_E.p_h(z0 = z_E)
                    time_tokens = self.vae.f_E.timeembedding(tidx)
                    x_rec = self.vae.f_D(z_H, c = ocs, zin = zin)
                    
                    ALL_ZE.append(z_E)
                    ALL_ZH.append(z_H)
                    if self.time_embedding:
                        ALL_time_tokens.append(time_tokens)
                    ALL_Z.append(z0)
                    ALL_ZH_Z0.append(z_H_z0)
                    # ALL_ZH.append(z_h)
                
                    # ALL_ZH.append(z_h)
                    # ALL_Z.append(prior_z)
                    ALL_C = None#.append(prior_c_logit if prior_c_logit is not None else prior_c)
                    ALL_Q = None #.append(q_code_mu)
                    
                    ALL_tidx.append(tidx)
        # On the last batch
        self.evaluation.recon_plot(x[0,:,:], x_rec[0,:,:], label = ["true", "recon"], epoch = str(epoch))
        # Memory intensive if the total sample size is large
        # ALL_ZE = torch.concatenate((ALL_ZE), dim = 0)
        ALL_ZE = torch.concatenate((ALL_ZE), dim = 0)
        
        if ALL_ZH is not None:
            ALL_ZH = torch.concatenate((ALL_ZH), dim = 0)
        if ALL_Z is not None:
            ALL_Z = torch.concatenate((ALL_Z), dim = 0)
            ALL_ZH_Z0 = torch.concatenate((ALL_ZH_Z0), dim = 0)
        # ALL_CE = torch.concatenate((ALL_CE), dim = 0)
        # ALL_C = torch.concatenate((ALL_C), dim = 0)
        # ALL_Q = torch.concatenate((ALL_Q), dim = 0)
        
        ALL_CGT = torch.concatenate((ALL_CGT), dim = 0)
        ALL_tidx = torch.concatenate((ALL_tidx), dim = 0)
        if self.time_embedding:
            ALL_time_tokens = torch.concatenate((ALL_time_tokens), dim = 0)
        # self.logger.info(f"Vali: Loss R:{Loss_R_vali.avg: .4f}, Loss Disc:{Loss_Disc_vali.avg: .4f}, Loss G: {Loss_G_vali.avg: .4f}")
        if self.vae.h is None:
            self.evaluation.vae_qualitative_analysis(ALL_Z, ALL_ZH, ALL_ZE,
                                                ALL_tidx,
                                                discrete = True if self.c_type == "discrete" else False,
                                                epoch = str(epoch))
            
        else:
            self.evaluation.nfvae_qualitative_analysis(ALL_Z, ALL_ZH, ALL_ZE, ALL_ZH_Z0,
                                                
                                                
                                                
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
        
        vae = src.VAE(dx=self.dx, dz=self.dz, dc=self.dc, window=self.window, 
                        d_model=self.d_model, num_heads=self.num_heads, z_projection=self.z_projection, 
                        D_projection=self.D_projection, time_emb=self.time_embedding, c_type=self.c_type, 
                        c_posterior_param=self.c_posterior_param, encoder_E=self.encoder_E, device=self.device,
                        supervised= self.supervised, kl_annealing=self.kl_annealing)
        
        print_model(vae, "VAE")
        self.vae, self.required_training = ut.load_model(vae, self.model_save_path, "VAE")
        self.vae.to(self.device)
        
    def get_optimizers(self):
        # lr constants 
        cont_R = 0.5
        cont_Disc = 0.1
        const_G = 1.

        opt_R, scheduler_R, wd_scheduler_R = ut.opt_constructor(self.scheduler,
                                                                        [self.vae.f_E, self.vae.f_D],
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
    