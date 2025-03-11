
import torch
import util_modules as ut
import src
import os
import tqdm 
import random
from pipelines import solver_base
from datasets.cmapss import load_CMAPSS
from datasets.circles import load_circle


class DataTrjPipeline(solver_base.Solver):
    def __init__(self, config, logger,
                 encoder = None,
                 shared_encoder_layer = None):
        super(DataTrjPipeline, self).__init__(config, logger)
        
        
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
        self.forecaster.train(False)
        return self.forecaster
    
    def train(self):
        self.logger.info("======================TRAINING BEGINS======================")
        self.training_log = ut.Logger(self.plots_save_path, 
                            self.his_save_path,
                            ('%d', 'epoch'),
                            ('%d', 'itr'),
                            
                            ('%.5f', 'loss_NLL')
                            )
        Loss_nll = ut.Value_averager()
        
        opt_fore = self.get_optimizers()
        
        vali_freq = 8  # total vali is vali_freq + 1
        vali_at = self.fore_epochs // vali_freq
        
        for epoch in tqdm(range(self.fore_epochs), desc = "Training data level trj: "):
            self.forecaster.train(True)
            for i, (x,y, _, tidx) in enumerate(self.training_data): # TODO 
                x = x.to(self.device)
                y = y.to(self.device)
                tidx = tidx.to(self.device)
                
                nll_loss = self.forecaster.Loss_NLL(y, x, tidx)
                opt_fore[0].zero_grad()
                nll_loss.backward()
                for m in reversed(opt_fore):
                    if m: m.step()
                
                # Records
                Loss_nll.update(nll_loss.item())
                
                self.training_log.log_into_csv_(epoch+1,
                                            i,
                                            Loss_nll.avg)
            self.logger.info(f"epoch[{epoch+1}/{self.fore_epochs}], Loss NLL:{Loss_nll.avg: .4f}")
            
            if vali_at > 0:
                if (self.validation) and (((epoch + 1) % vali_at) == 0):
                    self.vali(epoch+1)
        ...
        ...
        
        # Save the model at the end
        self.vali("evaluation")
        self.logger.info("Training a Data-trj is done. Saving the model ...")
        ut.save_model(self.forecaster, path_ = self.model_save_path, name = "trj")
        self.vali("full_test")
               
    def vali(self, epoch): # TODO
        self.forecaster.train(False)
        #TODO  --> need to get the raw x data for comparison also rather get un-windowed dictionary data for testing
        if epoch == "full_test":
            training_set, _, testing_set = self.full_t_v_t_sets
            N = len(training_set)
            inst = 10
            if inst > N:
                raise ValueError("S must be less than or equal to N to ensure unique values.")
            indx = random.sample(range(1, N + 1), inst)
            with torch.no_grad():
                for ii in tqdm(indx, desc = "Forecasting ... "):
                    x = training_set[ii]["hyper_lookbackwindowed_X"].to(self.device)
                    tidx = training_set[ii]["hyper_lookbackwindowed_time"].to(self.device)
                    ground_truth_x = training_set[ii]["hyper_windowed_X"].to(self.device)
                    
                    pred = self.forecaster(x, tidx)
      
        else:
            TRJ_X = []
            TRJ_PRED = []
            TRJ_TURE = []
            with torch.no_grad():
                if (self.val_data is None) or (epoch == "evaluation"):
                    data = self.testing_data
                else: # validation
                    data = self.val_data
                for i, (x,y, _, tidx) in tqdm(enumerate(data), desc = "Validation ..."): 
                    x = x.to(self.device)
                    tidx = tidx.to(self.device)
                    y = y.to(self.device)
                    
                    pred = self.forecaster(x, tidx)
                    TRJ_PRED.append(pred)
                    TRJ_TURE.append(y)
                    
                    if i == 0:
                        
                        x[0]              
                        RUL_PRED[0]
                        RUL_TRUE[0]
                        
                RUL_PRED = torch.concat((RUL_PRED), dim = 0)
                RUL_TRUE = torch.concat((RUL_TRUE), dim = 0)
                
                # TODO quantitative metric 
        
            # self.logger.info(f"Epoch ({epoch}) - RMSE: {RMSE: .4f}, SCORE: {SCORE: .4f}")
    def build_dataset(self):
        # Dataset instantiation
        if self.dataset == "circle": 
            raise NotImplementedError("")
            load_circle()
        else: 
            self.training_data, self.val_data, self.full_t_v_t_sets = load_CMAPSS(dataset = self.dataset,
                                                                    data_path = self.data_path,
                                                                    task = "data_trj",
                                                                    T = self.window,
                                                                    H = self.H,
                                                                    H_lookback = self.hyper_lookback,
                                                                    rectification = 125,
                                                                    batch_size = self.batch,
                                                                    normalize_rul = True,
                                                                    vis = False,
                                                                    logger = self.logger,
                                                                    plot = self.evaluation,
                                                                    valid_split= self.valid_split)
        
        self.ipe = len(self.training_data)
        
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
        
        if os.path.exists(os.path.join(self.model_save_path, 'rul_cp.pth')) and (encoder is None):
            print("A pre-trained rul predictor is available and being loaded ... ")
            shared_encoder_layer = src.SharedEncoder(dx=self.dx, dz=self.dz, window=self.window, d_model=self.d_model, 
                                          num_heads=self.num_heads,z_projection=self.z_projection, time_emb=self.time_embedding) # None
            prior_z = src.DiagonalGaussian(self.dz, mean = 0, var = 1, device=self.device)
            h = src.LatentFlow(self.dz, prior_z)
            f_E = src.LatentEncoder(dx=self.dx, dz=self.dz, window=self.window, d_model=self.d_model, 
                                          num_heads=self.num_heads, z_projection=self.z_projection, 
                                          time_emb=self.time_embedding, encoder_E=self.encoder_E, p_h=h, 
                                          shared_EC= True if shared_encoder_layer is not None else False) # TODO ARGUMENTS AND also how to deal with the shared part?
          
            forecaster = src.DataTrajectory(dz=self.dz, 
                                               # TODO
                                                pretrained_encoder=f_E, 
                                                shared_layer = shared_encoder_layer,
                                                device=self.device)  
        else:
            forecaster = src.DataTrajectory(dz=self.dz, 
                                               
                                                pretrained_encoder=encoder, 
                                                shared_layer = shared_encoder_layer,
                                                device=self.device)  
            
        print_model(forecaster, "Data Trajectory")
        self.forecaster, self.required_training = ut.load_model(forecaster, self.model_save_path, "trj")
        self.forecaster.to(self.device)
        
    def get_optimizers(self):
        opt_fore, scheduler_fore, wd_scheduler_fore = ut.opt_constructor(True, #self.scheduler,
                                                                [self.forecaster.predictor],
                                                                lr = self.lr_,
                                                                warm_up = int(self.fore_epochs* self.ipe * self.warm_up),
                                                                fianl_step = int(self.rul_epochs* self.ipe),
                                                                start_lr = self.start_lr,
                                                                ref_lr = self.ref_lr,
                                                                final_lr = self.final_lr,
                                                                start_wd = self.start_wd,
                                                                final_wd = self.final_wd,
                                                                
                                                                ft_lr_rate = 5e-2)
        return [opt_fore, scheduler_fore, wd_scheduler_fore]