
import torch
import util_modules as ut
import src
import os
from tqdm import tqdm
import random
from copy import deepcopy 
from pipelines import solver_base
from datasets.cmapss import load_CMAPSS
from datasets.circles import load_circle


class DataTrjPipeline(solver_base.Solver):
    def __init__(self, config, logger, trained_extractor):
        super(DataTrjPipeline, self).__init__(config, logger)
        
        self.surrogate = "decoder"
        self.build_dataset()
        self.build_model(trained_extractor)
        
    def __call__(self, validation = True):
        self.validation = validation
        if self.required_training:
            self.train()
        else:
            self.training_log = None
            # self.vali("evaluation")
            # self.vali("full_test")
            self.vis_on_full_seq()
            # self.vis_on_full_seq(specific_instance= [5,10,15,20], tt = [60, 80, 100, 120])
        self.forecaster.train(False)
        return self.forecaster
    
    def train(self):
        self.logger.info("======================TRAINING BEGINS======================")
        self.training_log = ut.Logger(self.plots_save_path, 
                            self.his_save_path,
                            ('%d', 'epoch'),
                            ('%d', 'itr'),
                            
                            ('%.5f', 'loss_NLL'),
                            ('%.5f', 'loss_NLL_z')
                            )
        Loss_nll = ut.Value_averager()
        Loss_nll_latent = ut.Value_averager()
        
        opt_fore = self.get_optimizers()
        
        vali_freq = 8  # total vali is vali_freq + 1
        vali_at = self.fore_epochs // vali_freq
        beta = 0.5
        for epoch in tqdm(range(self.fore_epochs), desc = "Training data level trj: "):
            self.forecaster.train(True)
            for i, (x,y, _, tidx) in enumerate(self.training_data): # TODO 
                x = x.to(self.device)
                y = y.to(self.device)
                tidx = tidx.to(self.device)
                
                # trained_extractor with decoder (or generator) is used as a surrogate  
                N, H, T, dx = y.shape 
                if self.time_embedding:
                    tidx_y = torch.arange(1, H + 1) * T  # Shape: (H,)
                    # Expand the shape to (N, H, 1)
                    tidx_y = tidx_y.repeat(N, 1).unsqueeze(-1) + tidx[:,-1:,:]
                    tidx_y = tidx_y.view(-1,1)
                else: tidx_y = None
                stationarized_X, stationarized_X_G = self.trained_extractor.stationarization(y.view(-1,T, dx), tidx_y)
                if stationarized_X_G is not None:
                    y = stationarized_X_G if self.surrogate == "generator" else stationarized_X
                else:
                    y = stationarized_X
                stationarized_y = y.view(N, H, T, dx).detach()
                
                nll_loss, nll_loss_latent = self.forecaster.Loss_trj(stationarized_y, x, tidx)
                total_loss = ((1- beta) * nll_loss) + (beta * nll_loss_latent)
                opt_fore[0].zero_grad()
                total_loss.backward()
                for m in reversed(opt_fore):
                    if m: m.step()
                
                # Records
                Loss_nll.update(nll_loss.item())
                Loss_nll_latent.update(nll_loss_latent.item())
                
                self.training_log.log_into_csv_(epoch+1,
                                            i,
                                            Loss_nll.avg,
                                            Loss_nll_latent.avg
                                            )
            self.logger.info(f"epoch[{epoch+1}/{self.fore_epochs}], Loss NLL:{Loss_nll.avg: .4f}, Loss latent :{Loss_nll_latent.avg: .4f}")
            
            # if vali_at > 0:
            #     if (self.validation) and (((epoch + 1) % vali_at) == 0):
            #         self.vali(epoch+1)
        ...
        ...
        self.vis_on_full_seq()
        # Save the model at the end
        # self.vali("evaluation")
        # self.logger.info("Training a Data-trj is done. Saving the model ...")
        ut.save_model(self.forecaster, path_ = self.model_save_path, name = "trj")
        # self.vali("full_test")
               
    def vali(self, epoch): # TODO
        self.forecaster.train(False)
        if epoch == "test":
            data = self.full_data[-1]
            gt_data = self.full_data[1]
            # original_data = self.original_data[1]
        else:
            data = self.full_data[-2]
            gt_data = self.full_data[0]
        TRJ_X = []
        TRJ_PRED = []
        TRJ_TURE = []
        
        with torch.no_grad():
            for key in tqdm(data, desc = "Forecasting ... "):
                # Dictionary wise full sequence
                x = data[key]["X"] # (L_seq, dx)
                tidx = data[key]["tidx"] # (L_seq, 1)
                
                # TODO: windowing
                
                # forward
                # pred, log_var = self.forecaster(input_window_X, input_window_tidx)
                
                # TODO window or format gt_data
                # gt_data[key]["X"]
                
                # TODO Some metrics to compute difference 
                
        # with torch.no_grad():
        #     data = self.val_data
        #     for i, (x,y, _, tidx) in tqdm(enumerate(data), desc = "Validation ..."): 
        #         x = x.to(self.device)
        #         tidx = tidx.to(self.device)
        #         y = y.to(self.device)
                
        #         pred = self.forecaster(x, tidx)
        #         TRJ_PRED.append(pred)
        #         TRJ_TURE.append(y)
                
        #         if i == 0:
                    
        #             x[0]              
        #             RUL_PRED[0]
        #             RUL_TRUE[0]
                    
        #     RUL_PRED = torch.concat((RUL_PRED), dim = 0)
        #     RUL_TRUE = torch.concat((RUL_TRUE), dim = 0)
            
            # TODO quantitative metric --> we need have ground truth TODO check how we should do this after vis 
            # TODO Calibrating the output prediction 
            
            
            # self.logger.info(f"Epoch ({epoch}) - RMSE: {RMSE: .4f}, SCORE: {SCORE: .4f}")
    def vis_on_full_seq(self, run = "test", epoch = "", specific_instance = [], tt = []):
        self.forecaster.train(False)
        if run == "test":
            data = self.full_data[-1]
            # original_data = self.original_data[1]
        elif run == "vali":
            data = self.full_data[-2]
            # original_data = self.original_data[0]
            
        # Starting time for eval 
        # Pick random 5 instances for vis
        if len(specific_instance) > 0:
            indx = specific_instance
        else:
            N = len(data)
            inst = 5
            if inst > N:
                raise ValueError("S must be less than or equal to N to ensure unique values.")
            indx = random.sample(range(1, N + 1), inst)
        
        with torch.no_grad():
            for ii in tqdm(indx, desc = "Forecasting ... "):
                # Dictionary wise full sequence
                x = data[ii]["X"] # (L_seq, dx)
                tidx = data[ii]["tidx"] # (L_seq, 1)
                if len(tt) > 0:
                    for t in tt:
                        self.full_forecasting(x, tidx, ii, epoch, t = t)
                else:
                    self.full_forecasting(x, tidx, ii, epoch)
                # total_len = x.shape[0]
                # min_len = (self.window * self.hyper_lookback) + (self.window * self.H)
                # if total_len < min_len + 50:
                #     t = 60
                # else:
                #     t = 100
                # if self.window * self.hyper_lookback > t:
                #         t = self.window * self.hyper_lookback 
                # # get lookback window sample
                # input_window_X = [torch.tensor(x[t-(self.window*(1+w)):t-(self.window*(w))], dtype = torch.float32) for w in reversed(range(self.hyper_lookback))]
                # input_window_X = torch.stack((input_window_X), dim = 0)
                # input_window_X = input_window_X.unsqueeze(0).to(self.device)

                # input_window_tidx = [torch.tensor(tidx[t-(self.window*(w)) -1],dtype = torch.long) for w in range(self.hyper_lookback)]
                # input_window_tidx = torch.stack((input_window_tidx), dim = 0) # W, 1
                # input_window_tidx = input_window_tidx.unsqueeze(0).to(self.device)  # 1, W, 1
                # pred, log_var = self.forecaster(input_window_X, input_window_tidx) # 1, H*T, dx
          
                # # stationarization
                # _, W, T, dx = input_window_X.shape 
                # stationarized_X, stationarized_X_G = self.trained_extractor.stationarization(input_window_X.view(-1,T, dx), tidx.view(-1,1)) # W, T, dx
                # if stationarized_X_G is not None:
                #     stationarized_GT = stationarized_X_G if self.surrogate == "generator" else stationarized_X
                # else:
                #     stationarized_GT = stationarized_X
                     
                # # uncertainty with 95% confidnnce interval
                # uncertainty_rate = torch.exp(0.5 * log_var) * 1.96
                
                # # Vis
                # pred = pred.view(-1, dx).detach().cpu().numpy() # H*T, 1
                # stationarized_GT = stationarized_GT.view(-1, dx).detach().cpu().numpy() # W*T, 1
                # x = x.detach().cpu().numpy()
                # uncertainty_rate = uncertainty_rate.view(-1, dx).detach().cpu().numpy()
       
                # self.evaluation.forecasting_plot(x, pred, stationarized_GT,
                #                                  t, int(t-(self.window*(self.hyper_lookback))), uncertainty_rate,
                #                                  title = str(ii), epoch=epoch)
    def full_forecasting(self, x, tidx, 
                         instance, epoch, t = None):
        total_len = x.shape[0]
        min_len = (self.window * self.hyper_lookback) + (self.window * self.H)
        if t == None:
            if total_len < min_len + 50:
                t = 60
            else:
                t = 100
        else: pass
        if self.window * self.hyper_lookback > t:
                t = self.window * self.hyper_lookback 
        # get lookback window sample
        input_window_X = [torch.tensor(x[t-(self.window*(1+w)):t-(self.window*(w))], dtype = torch.float32) for w in reversed(range(self.hyper_lookback))]
        input_window_X = torch.stack((input_window_X), dim = 0)
        input_window_X = input_window_X.unsqueeze(0).to(self.device)

        input_window_tidx = [torch.tensor(tidx[t-(self.window*(w)) -1],dtype = torch.long) for w in range(self.hyper_lookback)]
        input_window_tidx = torch.stack((input_window_tidx), dim = 0) # W, 1
        input_window_tidx = input_window_tidx.unsqueeze(0).to(self.device)  # 1, W, 1
        pred, log_var = self.forecaster(input_window_X, input_window_tidx) # 1, H*T, dx
    
        # stationarization
        _, W, T, dx = input_window_X.shape 
        stationarized_X, stationarized_X_G = self.trained_extractor.stationarization(input_window_X.view(-1,T, dx), tidx.view(-1,1)) # W, T, dx
        if stationarized_X_G is not None:
            stationarized_GT = stationarized_X_G if self.surrogate == "generator" else stationarized_X
        else:
            stationarized_GT = stationarized_X
                
        # uncertainty with 95% confidnnce interval
        uncertainty_rate = torch.exp(0.5 * log_var) * 1.96
        
        # Vis
        pred = pred.view(-1, dx).detach().cpu().numpy() # H*T, 1
        stationarized_GT = stationarized_GT.view(-1, dx).detach().cpu().numpy() # W*T, 1
        x = x.detach().cpu().numpy()
        uncertainty_rate = uncertainty_rate.view(-1, dx).detach().cpu().numpy()

        self.evaluation.forecasting_plot(x, pred, stationarized_GT,
                                            t, int(t-(self.window*(self.hyper_lookback))), uncertainty_rate,
                                            title = str(instance), epoch=epoch)
    def build_dataset(self):
        # Dataset instantiation
        if self.dataset == "circle": 
            raise NotImplementedError("")
            load_circle()
        else: 
            self.training_data, self.val_data, self.testing_data, self.full_data, self.normalizer = load_CMAPSS(dataset = self.dataset,
                                                                    data_path = self.data_path,
                                                                    task = "data_trj",
                                                                    T = self.window,
                                                                    H = self.H,
                                                                    H_lookback = self.hyper_lookback,
                                                                    rectification = 125,
                                                                    batch_size = self.fore_batch,
                                                                    normalize_rul = True,
                                                                    vis = False,
                                                                    logger = self.logger,
                                                                    plot = self.evaluation,
                                                                    valid_split= self.valid_split)
        
        self.ipe = len(self.training_data)
        
    def build_model(self, trained_extractor):
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
        
        # if os.path.exists(os.path.join(self.model_save_path, 'trj_cp.pth')):
        #     print("A pre-trained trj predictor is available and being loaded ... ")
        #     # shared_encoder_layer = src.SharedEncoder(dx=self.dx, dz=self.dz, window=self.window, d_model=self.d_model, 
        #     #                               num_heads=self.num_heads,z_projection=self.z_projection, time_emb=self.time_embedding) # None
        #     # prior_z = src.DiagonalGaussian(self.dz, mean = 0, var = 1, device=self.device)
        #     # h = src.LatentFlow(self.dz, prior_z)
        #     # f_E = src.LatentEncoder(dx=self.dx, dz=self.dz, window=self.window, d_model=self.d_model, 
        #     #                               num_heads=self.num_heads, z_projection=self.z_projection, 
        #     #                               time_emb=self.time_embedding, encoder_E=self.encoder_E, p_h=h, 
        #     #                               shared_EC= True if shared_encoder_layer is not None else False) # TODO ARGUMENTS AND also how to deal with the shared part?
          
        #     forecaster = src.DataTrajectory(dz=self.dz, 
        #                                        # TODO
        #                                         pretrained_encoder=deepcopy(self.trained_extractor.encoder), 
        #                                         shared_layer = deepcopy(self.trained_extractor.shared_encoder_layer),
        #                                         device=self.device)  
        # else:
        forecaster = src.DataTrajectory(dz=self.dz, dc=self.dc, dx=self.dx, 
                                        W=self.hyper_lookback, H =self.H, T=self.window, time_emb=self.time_embedding, 
                                        c_type=self.c_type, 
                                        pretrained_encoder=deepcopy(trained_extractor.f_E), 
                                        shared_layer = deepcopy(trained_extractor.shared_encoder_layers) if trained_extractor.shared_encoder_layers is not None else None,
                                        device=self.device,
                                        code = self.conditional)  
        
        print_model(forecaster, "Data Trajectory")
        self.forecaster, self.required_training = ut.load_model(forecaster, self.model_save_path, "trj")
        self.forecaster.to(self.device)
        
        self.trained_extractor = trained_extractor
        self.trained_extractor.to(self.device)
    def get_optimizers(self):
        opt_fore, scheduler_fore, wd_scheduler_fore = ut.opt_constructor(False, #self.scheduler,
                                                                [self.forecaster.predictor],
                                                                lr = self.trj_lr,
                                                                warm_up = int(self.fore_epochs* self.ipe * self.warm_up),
                                                                fianl_step = int(self.fore_epochs* self.ipe),
                                                                start_lr = self.start_lr,
                                                                ref_lr = self.ref_lr,
                                                                final_lr = self.final_lr,
                                                                start_wd = self.start_wd,
                                                                final_wd = self.final_wd,
                                                                
                                                                ft_lr_rate = 5e-2)
        return [opt_fore, scheduler_fore, wd_scheduler_fore]