import os
import time
import numpy as np
import torch
import torch.nn as nn

from tqdm import tqdm
from pipelines import solver_base
import src.ocir2
import util_modules as ut
import src
class OCIR2RlPipeline(solver_base.Solver):
    def __init__(self, config, logger):
        super(OCIR2RlPipeline, self).__init__(config, logger)
        
        
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
                            ('%.5f', 'loss_kl'),
                            ('%.5f', 'loss_MLE'),
                            ('%.5f', 'loss_recG'),
                            
                            ('%.5f', 'loss_realD'),
                            ('%.5f', 'loss_fakeD'),
                            
                            ('%.5f', 'loss_gen'),
                            ('%.5f', 'loss_Q'),
                            ('%.5f', 'loss_ccz'),
                            ('%.5f', 'loss_ccc'),
                            )
        
        Loss_R = ut.Value_averager()
        Loss_Disc = ut.Value_averager()
        Loss_G = ut.Value_averager()
        
        # In order to track the losses 
        loss_rec = ut.Value_averager()
        loss_kl = ut.Value_averager()
        loss_mle = ut.Value_averager()
        loss_recG = ut.Value_averager()
        
        loss_realD = ut.Value_averager()
        loss_fakeD = ut.Value_averager()
        
        loss_gen = ut.Value_averager()
        loss_Q = ut.Value_averager()
        loss_ccz = ut.Value_averager()
        loss_ccc = ut.Value_averager()
                
        opt_R, opt_Disc, opt_G = self.get_optimizers()
        vali_freq = 8  # total vali is vali_freq + 1
        vali_at = self.n_epochs // vali_freq
        # raise NotImplementedError("")
        self.vali("Initial")
        for epoch in tqdm(range(self.n_epochs), desc = "Training OCIR: "):
            self.ocir.train(True)
            if isinstance(self.ocir.prior_z, src.distributions.ContinuousCategorical):
                self.ocir.prior_z.step() 
            for i, (x,y, ocs, tidx) in enumerate(self.training_data): # TODO 
                x = x.to(self.device)
                tidx = tidx.to(self.device)

                # (1)
                loss_R, R = self.ocir.L_R(x, tidx, epoch = epoch+1)
                opt_R[0].zero_grad()
                
                # loss_REC, loss_KL, loss_MLE, loss_REC_G = R
                # loss_REC_G.backward(retain_graph = True)
                # ut.zeroout_gradient([self.ocir.f_E, self.ocir.f_C, self.ocir.shared_encoder_layers])
                # loss_vae = loss_REC+ loss_KL
                # loss_vae.backward()
                
                loss_R.backward()
                # loss_R.backward()
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
                
                # gen_loss, NLL_loss_Q, cc_loss_z, cc_loss_c = G
                # cc_loss = 0.2 *(cc_loss_z + (0.01 * cc_loss_c))
                # cc_loss.backward(retain_graph = True)
                # ut.zeroout_gradient([self.ocir.G]) # no grad in G wrt cc loss
                
                # gen_loss = gen_loss + NLL_loss_Q
                # gen_loss.backward()
                
                # for name, param in self.ocir.f_E.named_parameters():
                #     if param.grad is not None:
                #         print(f"Epoch {epoch},F_E-IN-G - {name} grad norm: {param.grad.norm().item()} \n")
                #     else:
                #         print("F_E-IN-G None \n")
                # torch.nn.utils.clip_grad_norm_(self.ocir.f_C.parameters(), max_norm=1.0)
                # for name, param in self.ocir.f_C.named_parameters():
                #     if param.grad is not None:
                #         print(f"Epoch {epoch},F_C-IN-G - {name} grad norm: {param.grad} \n")
                #     else:
                #         print("F_C-in-G None \n")
                loss_G.backward()
                for m in reversed(opt_G):
                    if m: m.step()
                

                # Records
                Loss_R.update(loss_R.item())
                Loss_Disc.update(loss_disc.item())
                Loss_G.update(loss_G.item())
                
                loss_rec.update(R[0].item())
                loss_kl.update(R[1].item())
                loss_mle.update(R[2].item())
                loss_recG.update(R[3].item())
                
                loss_realD.update(D[0].item())
                loss_fakeD.update(D[1].item())
                
                loss_gen.update(G[0].item())
                loss_Q.update(G[1].item())
                loss_ccz.update(G[2].item())
                loss_ccc.update(G[3].item())
                self.training_log.log_into_csv_(epoch+1,
                                            i,
                                            loss_rec.avg,
                                            loss_kl.avg,
                                            loss_mle.avg,
                                            loss_recG.avg,
                                            
                                            loss_realD.avg,
                                            loss_fakeD.avg,
                                            
                                            loss_gen.avg,
                                            loss_Q.avg,
                                            loss_ccz.avg,
                                            loss_ccc.avg,
                                            )
            self.logger.info(f"epoch[{epoch+1}/{self.n_epochs}], Loss R:{Loss_R.avg: .4f}, Loss Disc:{Loss_Disc.avg: .4f}, Loss G: {Loss_G.avg: .4f}")
            
            if (self.validation) and (((epoch + 1) % vali_at) == 0):
                self.vali(epoch+1)
        # Save the model at the end
        self.logger.info("Training a OCIR is done. Saving the model ...")
        ut.save_model(self.ocir, path_ = self.model_save_path, name = "OCIR")
        
    def vali(self, epoch): # TODO eval
        # Note that it is "not necessary" as we are doing unsupervised learning. 
        # We mainly check whether the objective functions converge or not.
        # Loss_R_vali = ut.Value_averager()
        # Loss_Disc_vali = ut.Value_averager()
        # Loss_G_vali = ut.Value_averager()
        self.ocir.train(False)
        
        ALL_ZE = []
        ALL_Z = []
        ALL_ZH = []
        
        ALL_CE = []
        ALL_C = []
        ALL_Q = []
        ALL_CGT = []
        
        ALL_tidx = []
        for i, (x,_, ocs, tidx) in enumerate(self.val_data): 
            x = x.to(self.device)
            tidx = tidx.to(self.device)
            # Inference
            if self.ocir.shared_encoder_layers is not None:
                h = self.ocir.shared_encoder_layers(x, tidx)
                hc = h
                if (self.z_projection == "spc") or (self.z_projection == "seq"):
                    hc = hc[:,1:,:]
                if self.time_embedding:
                    hc = hc[:,:-1,:]
            else: 
                h = x
                hc = x
            z_E = self.ocir.f_E.encoding(h, tidx)
            c_E = self.ocir.f_C.inference(hc, logits = True)
            x_rec = self.ocir.f_D(z_E, c = c_E, zin = None)
            # Generation
            X_gen, set_latent_samples, log_det = self.ocir.G.generation(x.shape[0])
            q_code_mu = self.ocir.Q.inference(X_gen, logits = True)
            
            z_h, prior_z, prior_c, prior_c_logit = set_latent_samples

            ALL_ZE.append(z_E)
            ALL_Z.append(prior_z)
            ALL_ZH.append(z_h)
            
            ALL_C.append(prior_c_logit if prior_c_logit is not None else prior_c)
            ALL_CE.append(c_E)
            ALL_Q.append(q_code_mu)
            ALL_CGT.append(ocs)
            
            ALL_tidx.append(tidx)
        self.evaluation.recon_plot(x[0,:,:], x_rec[0,:,:], label = ["true", "recon"], epoch = str(epoch))
        # Memory intensive if the total sample size is large
        ALL_ZE = torch.concatenate((ALL_ZE), dim = 0)
        ALL_Z = torch.concatenate((ALL_Z), dim = 0)
        ALL_ZH = torch.concatenate((ALL_ZH), dim = 0)
        ALL_CE = torch.concatenate((ALL_CE), dim = 0)
        ALL_C = torch.concatenate((ALL_C), dim = 0)
        ALL_Q = torch.concatenate((ALL_Q), dim = 0)
        ALL_CGT = torch.concatenate((ALL_CGT), dim = 0)
        ALL_tidx = torch.concatenate((ALL_tidx), dim = 0)
        # self.logger.info(f"Vali: Loss R:{Loss_R_vali.avg: .4f}, Loss Disc:{Loss_Disc_vali.avg: .4f}, Loss G: {Loss_G_vali.avg: .4f}")
        max_num = -1
        self.evaluation.qualitative_analysis(ALL_Z[:max_num], ALL_ZH[:max_num], ALL_ZE[:max_num],
                                             ALL_C[:max_num], ALL_CE[:max_num], ALL_CGT[:max_num], ALL_Q[:max_num],
                                             ALL_tidx[:max_num],
                                             discrete = True if self.c_type == "discrete" else False,
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
        
        ocir = src.OCIR2(dx=self.dx, dz=self.dz, dc=self.dc, window=self.window, 
                        d_model=self.d_model, num_heads=self.num_heads, z_projection=self.z_projection, 
                        D_projection=self.D_projection, time_emb=self.time_embedding, c_type=self.c_type, 
                        c_posterior_param=self.c_posterior_param, encoder_E=self.encoder_E, device=self.device)
        
        print_model(ocir, "OCIR2")
        self.ocir = ut.load_model(ocir, self.model_save_path, "OCIR2")
        self.ocir.to(self.device)
        
    def get_optimizers(self):
        # lr constants 
        cont_R = 0.5
        cont_Disc = 0.5
        const_G = 1.
        opt_R, scheduler_R, wd_scheduler_R = ut.opt_constructor(self.scheduler,
                                                                        [self.ocir.f_E,  self.ocir.f_C, self.ocir.f_D, self.ocir.G, self.ocir.shared_encoder_layers], # , self.ocir.h
                                                                       lr = self.lr_ * cont_R,
                                                                        warm_up = int(self.n_epochs* self.ipe * self.warm_up),
                                                                        fianl_step = int(self.n_epochs* self.ipe),
                                                                        start_lr = self.start_lr,
                                                                        ref_lr = self.ref_lr,
                                                                        final_lr = self.final_lr,
                                                                        start_wd = self.start_wd,
                                                                        final_wd = self.final_wd)
        opt_Disc, scheduler_Disc, wd_scheduler_Disc = ut.opt_constructor(self.scheduler, 
                                                                        [self.ocir.D],
                                                                        lr = self.lr_ * cont_Disc,
                                                                        warm_up = int(self.n_epochs* self.ipe * self.warm_up),
                                                                        fianl_step = int(self.n_epochs* self.ipe),
                                                                        start_lr = self.start_lr,
                                                                        ref_lr = self.ref_lr,
                                                                        final_lr = self.final_lr,
                                                                        start_wd = self.start_wd,
                                                                        final_wd = self.final_wd)
        opt_G, scheduler_G, wd_scheduler_G = ut.opt_constructor(self.scheduler,
                                                                       [self.ocir.f_E, self.ocir.f_C, self.ocir.G, self.ocir.Q, self.ocir.shared_encoder_layers], #, self.ocir.h
                                                                       lr = self.lr_ * const_G,
                                                                       warm_up = int(self.n_epochs* self.ipe * self.warm_up),
                                                                        fianl_step = int(self.n_epochs* self.ipe),
                                                                        start_lr = self.start_lr,
                                                                        ref_lr = self.ref_lr,
                                                                        final_lr = self.final_lr,
                                                                        start_wd = self.start_wd,
                                                                        final_wd = self.final_wd)
        # for opt in [opt_R, opt_Disc, opt_G]:
        #     for i, param_group in enumerate(opt.param_groups):
        #         self.logger.info(f"Group {i}: {param_group}")
        return [opt_R, scheduler_R, wd_scheduler_R], \
               [opt_Disc, scheduler_Disc, wd_scheduler_Disc], \
               [opt_G, scheduler_G, wd_scheduler_G]     
    