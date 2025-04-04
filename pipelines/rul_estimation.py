import os
import util_modules as ut
import src
import torch
import random
from copy import deepcopy
from tqdm import tqdm

from pipelines import solver_base #, VAEPipeline, RlPipeline

class RulPipeline(solver_base.Solver):
    def __init__(self, config, 
                 logger,
                 encoder = None,
                 shared_encoder_layer = None):
        super(RulPipeline, self).__init__(config, logger)
        
        self.build_dataset()
        self.build_model(encoder, shared_encoder_layer)
        
    def __call__(self, validation = True):
        self.validation = validation
        if self.required_training:
            self.train()
        else:
            self.training_log = None
            self.vali("evaluation")
            self.vali("full_test")
            self.vali("full_test", [5, 10, 15, 20])
        self.rul_predictor.train(False)
        return self.rul_predictor
    def train(self):
        self.logger.info("======================TRAINING BEGINS======================")
        
        self.training_log = ut.Logger(self.plots_save_path, 
                            self.his_save_path,
                            ('%d', 'epoch'),
                            ('%d', 'itr'),
                            
                            ('%.5f', 'loss_mse')
                            )
        Loss_MSE = ut.Value_averager()
        
        opt_rul = self.get_optimizers()
        
        vali_freq = 8  # total vali is vali_freq + 1
        vali_at = self.rul_epochs // vali_freq
        
        for epoch in tqdm(range(self.rul_epochs), desc = "Training RUL estimator: "):
            self.rul_predictor.train(True)
            for i, (x,y, _, tidx) in enumerate(self.training_data): # TODO 
                x = x.to(self.device)
                y = y.to(self.device)
                tidx = tidx.to(self.device)

                loss_rul = self.rul_predictor.Loss_RUL(y, x, tidx)
                opt_rul[0].zero_grad()
                loss_rul.backward()
                for m in reversed(opt_rul):
                    if m: m.step()
                
                # Records
                Loss_MSE.update(loss_rul.item())
                
                self.training_log.log_into_csv_(epoch+1,
                                            i,
                                            Loss_MSE.avg)
            self.logger.info(f"epoch[{epoch+1}/{self.rul_epochs}], Loss RUL:{Loss_MSE.avg: .4f}")
            
            if vali_at > 0:
                if (self.validation) and (((epoch + 1) % vali_at) == 0):
                    self.vali(epoch+1)
        ...
        ...
        
        # Save the model at the end
        self.vali("evaluation")
        self.logger.info("Training a RUL is done. Saving the model ...")
        ut.save_model(self.rul_predictor, path_ = self.model_save_path, name = "rul")
        self.vali("full_test")
    def vali(self, epoch, specific_instance = []):
        self.rul_predictor.train(False)
        RUL_PRED = dict()
        error = []
        if epoch == "full_test":
            with torch.no_grad():
                # Compute per each machine instance
                for i, key in tqdm(enumerate(self.full_test_set), desc = "Full testset evaluation ..."): 
                    x, y, ocs, tidx = self.full_test_set[key]["X"], self.full_test_set[key]["Y"], \
                                        self.full_test_set[key]["ocs"], self.full_test_set[key]["tidx"]
                    x = x.to(self.device)
                    tidx = tidx.to(self.device)
                    y = y.to(self.device)
                    
                    rul = self.rul_predictor(x, tidx)
                    RUL_PRED[key] = rul
                    _, _, ERRORS = ut.RUL_metric(pred=rul, target = y)
                    error.append(ERRORS.mean())
            min_value = min(error)
            min_index = error.index(min_value)
            
            if len(specific_instance) > 0:
                for ii in indx:
                    self.evaluation.rul_plot(RUL_PRED[ii].detach().cpu().numpy(), 
                                                self.full_test_set[ii]["Y"].detach().cpu().numpy(),
                                                str(ii))
            else:
                N = len(RUL_PRED)
                inst = 4
                if inst > N:
                    raise ValueError("S must be less than or equal to N to ensure unique values.")
                indx = random.sample(range(1, N + 1), inst)

                for ii in indx:
                    self.evaluation.rul_plot(RUL_PRED[ii].detach().cpu().numpy(), 
                                            self.full_test_set[ii]["Y"].detach().cpu().numpy(),
                                            str(ii))
            
                
        else:
            RUL_PRED = []
            RUL_TRUE = []
            with torch.no_grad():
                if (self.val_data is None) or (epoch == "evaluation"):
                    data = self.testing_data
                else:
                    data = self.val_data
                for i, (x,y, _, tidx) in enumerate(data): 
                    x = x.to(self.device)
                    tidx = tidx.to(self.device)
                    y = y.to(self.device)
                    
                    rul = self.rul_predictor(x, tidx)
                
                    RUL_PRED.append(rul) # TODO check the shape --> (N, 1)
                    RUL_TRUE.append(y)
                RUL_PRED = torch.concat((RUL_PRED), dim = 0)
                RUL_TRUE = torch.concat((RUL_TRUE), dim = 0)
                
                RMSE, SCORE, ERRORS = ut.RUL_metric(pred=RUL_PRED, target = RUL_TRUE)
                if (epoch == "evaluation") and self.training_log:
                    self.training_log.savevalues(ERRORS) # save to csv
                
            self.logger.info(f"Epoch ({epoch}) - RMSE: {RMSE: .4f}, SCORE: {SCORE: .4f}")
        
    def build_model(self, encoder = None, shared_encoder_layer = None):
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
        
        rul_predictor = src.RulEstimator(dz=self.dz, 
                                    pretrained_encoder=deepcopy(encoder.f_E), 
                                    shared_layer = deepcopy(encoder.shared_encoder_layers) if encoder.shared_encoder_layers is not None else None,
                                    device=self.device)  
        
        # if os.path.exists(os.path.join(self.model_save_path, 'rul_cp.pth')) and (encoder is None):
        #     print("A pre-trained rul predictor is available and being loaded ... ")
        #     shared_encoder_layer = src.SharedEncoder(dx=self.dx, dz=self.dz, window=self.window, d_model=self.d_model, 
        #                                   num_heads=self.num_heads,z_projection=self.z_projection, time_emb=self.time_embedding) # None
        #     prior_z = src.DiagonalGaussian(self.dz, mean = 0, var = 1, device=self.device)
        #     h = src.LatentFlow(self.dz, prior_z)
        #     f_E = src.LatentEncoder(dx=self.dx, dz=self.dz, window=self.window, d_model=self.d_model, 
        #                                   num_heads=self.num_heads, z_projection=self.z_projection, 
        #                                   time_emb=self.time_embedding, encoder_E=self.encoder_E, p_h=h, 
        #                                   shared_EC= True if shared_encoder_layer is not None else False) # TODO ARGUMENTS AND also how to deal with the shared part?
          
        #     rul_predictor = src.RulEstimator(dz=self.dz, 
        #                                     pretrained_encoder=f_E, 
        #                                     shared_layer = shared_encoder_layer,
        #                                     device=self.device)  
        #     #load here 
        # else:
        #     rul_predictor = src.RulEstimator(dz=self.dz, 
        #                             pretrained_encoder=encoder, 
        #                             shared_layer = shared_encoder_layer,
        #                             device=self.device)  
            
        print_model(rul_predictor, "RUL predictior")
        self.rul_predictor, self.required_training = ut.load_model(rul_predictor, self.model_save_path, "rul")
        self.rul_predictor.to(self.device)
        
    def get_optimizers(self):
        opt_RUL, scheduler_RUL, wd_scheduler_RUL = ut.opt_constructor(False, #self.scheduler,
                                                                [self.rul_predictor.shared_layer, self.rul_predictor.encoder, self.rul_predictor.regressor],
                                                                # [self.rul_predictor.regressor],
                                                                lr = self.rul_lr,
                                                                warm_up = int(self.rul_epochs* self.ipe * self.warm_up),
                                                                fianl_step = int(self.rul_epochs* self.ipe),
                                                                start_lr = self.start_lr,
                                                                ref_lr = self.ref_lr,
                                                                final_lr = self.final_lr,
                                                                start_wd = self.start_wd,
                                                                final_wd = self.final_wd,
                                                                
                                                                ft_lr_rate = 1e-1)
        return [opt_RUL, scheduler_RUL, wd_scheduler_RUL]