import os
import time
import numpy as np
import torch
import torch.nn as nn

from tqdm import tqdm
from pipelines import solver_base
import util_modules as ut
import src
class InfoGANPipeline(solver_base.Solver):
    def __init__(self, config, logger):
        super(InfoGANPipeline, self).__init__(config, logger)
        
        
        self.build_dataset()
        self.build_model()
    def __call__(self, validation = True):
        self.validation = validation
        if self.required_training:
            self.train()
        else:
            self.vali("validation")
        self.infogan.train(False)
        return self.infogan
    
    def train(self):
        self.logger.info("======================TRAINING BEGINS======================")
        
        self.training_log = ut.Logger(self.plots_save_path, 
                            self.his_save_path,
                            ('%d', 'epoch'),
                            ('%d', 'itr'),
                            
                            ('%.5f', 'loss_realD'),
                            ('%.5f', 'loss_fakeD'),
                            
                            ('%.5f', 'loss_gen'),
                            ('%.5f', 'loss_Q'),
                            ('%.5f', 'logdet')
                            )
        
        Loss_Disc = ut.Value_averager()
        Loss_G = ut.Value_averager()
        
        # In order to track the losses 
        
        loss_realD = ut.Value_averager()
        loss_fakeD = ut.Value_averager()
        
        loss_gen = ut.Value_averager()
        loss_Q = ut.Value_averager()
        loss_logdet = ut.Value_averager()
        
        opt_Disc, opt_G = self.get_optimizers()
        vali_freq = 8  # total vali is vali_freq + 1
        vali_at = self.n_epochs // vali_freq
        self.vali("Initial")
        for epoch in tqdm(range(self.n_epochs), desc = "Training infogan: "):
            self.infogan.train(True)
            for i, (x,y, ocs, tidx) in enumerate(self.training_data): # TODO
                pass 
                x = x.to(self.device)
                tidx = tidx.to(self.device)
                    
                # (2)
                loss_disc, D = self.infogan.Loss_Discriminator(x)
                opt_Disc[0].zero_grad()
                loss_disc.backward()
                for m in reversed(opt_Disc):
                    if m: m.step()
                
                # (3)
                loss_G, G = self.infogan.Loss_Generator(x, epoch = epoch+1)
                opt_G[0].zero_grad()
                loss_G.backward()
                if self.infogan.shared_net is not None:
                    ut.zeroout_gradient([self.infogan.shared_net])
                ii = 0
                for m in reversed(opt_G):
                    if m: m.step()

                # Records
                Loss_Disc.update(loss_disc.item())
                Loss_G.update(loss_G.item())
     
                loss_realD.update(D[0].item())
                loss_fakeD.update(D[1].item())
                
                loss_gen.update(G[0].item())
                loss_Q.update(G[1].item())
                loss_logdet.update(G[2].item())
                 
                self.training_log.log_into_csv_(epoch+1,
                                            i,
                                            
                                            loss_realD.avg,
                                            loss_fakeD.avg,
                                            
                                            loss_gen.avg,
                                            loss_Q.avg,
                                            loss_logdet.avg
                                            )
            self.logger.info(f"epoch[{epoch+1}/{self.n_epochs}], Loss Disc:{Loss_Disc.avg: .4f}, Loss G: {Loss_G.avg: .4f}, Loss logdet: {loss_logdet.avg: .4f}")
            
            if (self.validation) and (((epoch + 1) % vali_at) == 0):
                self.vali(epoch+1)
        # Save the model at the end
        self.logger.info("Training a infogan is done. Saving the model ...")
        ut.save_model(self.infogan, path_ = self.model_save_path, name = "infogan")
        
    def vali(self, epoch): # TODO eval
        self.infogan.train(False)
        
        ALL_ZE = None
        ALL_Z = []
        ALL_ZH = []
        
        ALL_CE = None
        ALL_C = []
        ALL_Q = []
        ALL_CGT = []
        
        ALL_tidx = []
        with torch.no_grad():
            for i, (x,_, ocs, tidx) in enumerate(self.val_data): 
                x = x.to(self.device)
                tidx = tidx.to(self.device)

                # Generation
                X_gen, set_latent_samples, _ = self.infogan.G.generation(x.shape[0]) # TODO vis the output generation
                q_code_mu = self.infogan.Q.inference(X_gen, logits = True)
                
                z_h, prior_z, prior_c, prior_c_logit = set_latent_samples

                if prior_z == None:
                    ALL_Z.append(z_h)
                    # ALL_ZH.append(z_h)
                    ALL_ZH = None
                else:
                    ALL_ZH.append(z_h)
                    ALL_Z.append(prior_z)
                ALL_C.append(prior_c_logit if prior_c_logit is not None else prior_c)
                ALL_Q.append(q_code_mu)
                ALL_CGT.append(ocs)
                
                ALL_tidx.append(tidx)
                
        self.evaluation.recon_plot(x[0,:,:], X_gen[0,:,:], label = ["true", "gen"], epoch = str(epoch))
        
        # Memory intensive if the total sample size is large
        ALL_Z = torch.concatenate((ALL_Z), dim = 0)
        if ALL_ZH is not None:
            ALL_ZH = torch.concatenate((ALL_ZH), dim = 0)
        ALL_C = torch.concatenate((ALL_C), dim = 0)
        ALL_Q = torch.concatenate((ALL_Q), dim = 0)
        ALL_CGT = torch.concatenate((ALL_CGT), dim = 0)
        ALL_tidx = torch.concatenate((ALL_tidx), dim = 0)

        self.evaluation.info_qualitative_analysis(ALL_Z, ALL_ZH, ALL_ZE,
                                             ALL_C, ALL_CE, ALL_CGT, ALL_Q,
                                             ALL_tidx,
                                             discrete = True if self.c_type == "discrete" else False,
                                             discreteuniform= isinstance(self.infogan.prior_c, src.distributions.DiscreteUniform),
                                             epoch = str(epoch))
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
        
        infogan = src.InfoGAN(dx=self.dx, dz=self.dz, dc=self.dc, window=self.window, 
                        d_model=self.d_model, num_heads=self.num_heads, z_projection=self.z_projection, 
                        D_projection=self.D_projection, time_emb=self.time_embedding, c_type=self.c_type, 
                        c_posterior_param=self.c_posterior_param, encoder_E=self.encoder_E, device=self.device)
        
        print_model(infogan, "infogan")
        self.infogan, self.required_training = ut.load_model(infogan, self.model_save_path, "infogan")
        self.infogan.to(self.device)
        
    def get_optimizers(self):
        # lr constants 
        cont_R = 0.5
        cont_Disc = 0.1
        const_G = 1.

        opt_Disc, scheduler_Disc, wd_scheduler_Disc = ut.opt_constructor(self.scheduler, 
                                                                        [self.infogan.D],
                                                                        lr = self.lr_ * cont_Disc,
                                                                        warm_up = int(self.n_epochs* self.ipe * self.warm_up),
                                                                        fianl_step = int(self.n_epochs* self.ipe),
                                                                        start_lr = self.start_lr,
                                                                        ref_lr = self.ref_lr,
                                                                        final_lr = self.final_lr,
                                                                        start_wd = self.start_wd,
                                                                        final_wd = self.final_wd)
        opt_G, scheduler_G, wd_scheduler_G = ut.opt_constructor(self.scheduler,
                                                                       [self.infogan.G, self.infogan.Q],
                                                                       lr = self.lr_ * const_G,
                                                                       warm_up = int(self.n_epochs* self.ipe * self.warm_up),
                                                                        fianl_step = int(self.n_epochs* self.ipe),
                                                                        start_lr = self.start_lr,
                                                                        ref_lr = self.ref_lr,
                                                                        final_lr = self.final_lr,
                                                                        start_wd = self.start_wd,
                                                                        final_wd = self.final_wd)

        return [opt_Disc, scheduler_Disc, wd_scheduler_Disc], \
               [opt_G, scheduler_G, wd_scheduler_G]     
    