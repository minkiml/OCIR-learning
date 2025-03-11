import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from src.blocks import src_utils, SharedEncoder, transformers
from src.modules import encoders

class Regressor(nn.Module):
    ''' 
    Wrapper class for a full rul estimator (trained invariant encoder + a regressor)
    '''
    def __init__(self, dz):
        super(Regressor, self).__init__()
        self.linear = src_utils.Linear(dz, 1, bias = False)
    
    def forward(self, z):
        return self.linear(z)
class RulEstimator(nn.Module):
    ''' 
    Wrapper class for a full rul estimator (trained invariant encoder + a regressor)
    '''
    def __init__(self, dz, 
                    pretrained_encoder:encoders.LatentEncoder,
                    shared_layer:SharedEncoder, 
                    device): 
        super(RulEstimator, self).__init__()
        self.device = device
        
        self.shared_layer = shared_layer
        self.encoder = pretrained_encoder    
        self.regressor = Regressor(dz)
        
    def forward(self, x, tidx = None):
        if self.shared_layer is not None:
            x = self.shared_layer(x, tidx)
            
        z = self.encoder.encoding(x, tidx)
        
        # z, log_var, _ = self.encoder(x, tidx)
        # z, _, _ = self.encoder.reparameterization_NF(z, log_var)
        rul = self.regressor(z)
        return rul
    
    def Loss_RUL(self, true_rul, x, tidx = None):
        rul_pred = self.forward(x, tidx)

        # TODO CHECK the shape of the two args. 
        return F.mse_loss(rul_pred, true_rul)  # l1 loss
        
class Forecaster(nn.Module):
    ''' 
    Simply use a transformer-encoder-based forcaster, 
    but any other SOTA model would improve the performance.
    '''
    def __init__(self, dz, d_out, 
                 W, H, T,
                 time_emb, 
                 d_model = 128):
        super(Forecaster, self).__init__()
        self.time_emb = time_emb
        self.out_length = H * T
        self.d_out = d_out
        
        self.up_projection = src_utils.Linear(dz, d_model, bias = False)
        # TODO fixed code input !
        if self.time_emb:
            self.time_embedding = nn.Embedding(num_embeddings= 550, embedding_dim = d_model)
            # src_utils.init_embedding(self.time_embedding)
        self.pos_enc = transformers.SinCosPositionalEncoding(d_model, W + 1)
        forecaster = [transformers.TransformerEncoderBlock(embed_dim = d_model, num_heads = self.num_heads,
                                                        ff_hidden_dim = int(d_model * 3), dropout = 0.10,
                                                        prenorm =True) for _ in range(self.depth)]
        self.forecaster = nn.ModuleList(forecaster)
        
        
        self.wide_linear = transformers.Conv1by1(d_model * W, d_out * H * T * 2) # transformers.Conv1by1(dx, d_model)

    def forward(self, z, tidx = None):
        N, W, _ = z.shape
        z = self.up_projection(z)
        
        # Time index embeddings
        if self.time_emb:
            if tidx is not None: 
                time_token = self.time_embedding(tidx) 
            else:
                time_token = self.time_embedding(torch.zeros((N,1), dtype=torch.long).to(z.device)) # zero token
            z = torch.cat((time_token,z), dim = 1)
        # Positional encoding
        z = self.pos_enc(z)
        
        # transformer
        for layer in self.forecaster:
            z = layer(z)
        
        z = torch.flatten(z, 1, -1)
        y = self.wide_linear(z) 
        mu, log_var = torch.chunk(y, 2, dim=-1)
        mu, log_var = mu.view(N,self.out_length,self.d_out), log_var.view(N,self.out_length,self.d_out) 
        
        return mu, log_var

class DataTrajectory(nn.Module):
    def __init__(self, dz, d_out, W, H, T, time_emb, 
                    pretrained_encoder:encoders.LatentEncoder,
                    shared_layer:SharedEncoder, 
                    device): 
        super(DataTrajectory, self).__init__()
        self.device = device
        self.T = T
        self.H = H
        self.W = W
        self.time_emb = time_emb
        
        self.shared_layer = shared_layer
        self.encoder = pretrained_encoder 
        
        self.predictor = Forecaster(dz, d_out, 
                                    W, H, T, 
                                    time_emb)
        
    def forward(self, x, tidx = None):
        z, z_logvar = self.encode(x, tidx)
        y, log_var = self.predictor(z, tidx)
        return y, log_var
    
    def encode(self, x, tidx = None, dist = False):
        N, W, T, dx = x.shape
        x = x.view(-1, T, dx)
        if self.time_emb:
            tidx = tidx.view(-1, 1)
        if self.shared_layer is not None:
            x = self.shared_layer(x, tidx)
        
        if not dist:
            z = self.encoder.encoding(x, tidx)
            z = z.view(N,W,-1)
            z_log_var = None
        else: z, z_log_var, _ = self.encoder(x, tidx)
        return z, z_log_var
    
    def Loss_trj(self, y, x, tidx = None):
        N = y.shape[0]
        mu, log_var = self.forward(x, tidx)

        if mu is None:
            mu = self.mean
        
        log_std = 0.5 * log_var  # log(std) = 0.5 * log_var
        y = y.view(N,self.H * self.T, -1)
        
        diff = y - mu
        log_prob = -0.5 * (diff / torch.exp(log_std)) ** 2  # Quadratic term
        log_prob -= 0.5 * np.log(2 * np.pi)  # Constant term
        log_prob -= log_std  
        log_prob = log_prob.sum(dim=-1)  # Sum over dimensions
        loss_nll = -torch.mean(log_prob)
        
        if self.time_emb:
            tidx_y = torch.arange(1, self.H + 1) * self.T  # Shape: (H,)
            # Expand the shape to (N, H, 1)
            tidx_y = tidx_y.repeat(N, 1).unsqueeze(-1) + tidx[:,-1:,:]
        else: tidx_y = None
        
        mu = mu.view(N,self.H, self.T, -1)
        z_pred, z_logvar = self.encode(mu, tidx_y) # z of shape (N, H, dz)
        z_y = self.encode(y, tidx_y) # z of shape (N, H, dz)
        
        # TODO mse or nll 
        zlog_std = 0.5 * z_logvar
        diff = z_y - z_pred
        log_prob = -0.5 * (diff / torch.exp(zlog_std)) ** 2  # Quadratic term
        log_prob -= 0.5 * np.log(2 * np.pi)  # Constant term
        log_prob -= zlog_std  
        log_prob = log_prob.sum(dim=-1)  # Sum over dimensions
        loss_nll_z = -torch.mean(log_prob)
        
        return loss_nll, loss_nll_z