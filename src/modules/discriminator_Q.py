'''
Discriminator, D

Approximate posterior, Q

'''

import torch.nn.functional as F
import torch
import torch.nn as nn

from src.blocks import transformers, src_utils 
from src.modules import distributions


class Discriminator(nn.Module): # bunch of encoder layers
    def __init__(self, dx:int, window:int, 
                 d_model:int, num_heads:int, D_projection:str,
                 shared_layer = None):
        super(Discriminator, self).__init__()

        self.depth = 2
        self.num_heads = num_heads
        self.D_projection = D_projection
        self.shared_layer = shared_layer
        
        if shared_layer is None:
            self.pos_enc = transformers.SinCosPositionalEncoding(d_model, window + 1)
            self.fE_projection = transformers.Conv1by1(dx, d_model)
            discriminator_layers = [transformers.TransformerEncoderBlock(embed_dim = d_model, num_heads = self.num_heads,
                                                                        ff_hidden_dim = int(d_model * 3), dropout = 0.15,
                                                                        prenorm = True) for _ in range(self.depth)]
            self.discriminator_layers = nn.ModuleList(discriminator_layers)
        else:
            self.discriminator_layers = nn.Sequential(src_utils.Linear(d_model, d_model),
                                                      nn.LeakyReLU(0.2),
                                                      nn.Linear(d_model, d_model),
                                                      nn.LeakyReLU(0.2))
        # (N, L, C) -> (N, C)
        if D_projection == "aggregation": 
            self.aggregation = transformers.Aggregation(d_model, method = 'weighted')
            
        elif D_projection == "spc" and (shared_layer is None):
            # otherwise, a BERT-style special token (namely compressive token) is done 
            self.score_token = nn.Parameter(torch.randn(1,1,d_model) * 0.02)
            
        elif D_projection == "rnn":
            self.aggregation = src_utils.rnn_aggregation(d_model, d_model)
        
        elif D_projection == "None":
            self.aggregation = nn.Identity() 
        
        self.regressor = src_utils.Linear(d_model, 1)
        
    def forward(self, x):
        # x (N, L, c)
        N, L, c = x.shape
        x_emb = x
        if self.shared_layer is None:
            x_emb = self.fE_projection(x_emb)
            if (self.D_projection == "spc"):
                score_token = self.score_token.expand(N,-1,-1)
                x_emb = torch.cat((score_token, x_emb), dim = 1)    
                            
            # positional encoding
            x_emb = self.pos_enc(x_emb)
            for layer in self.discriminator_layers:
                x_emb = layer(x_emb)
        else:
            x_emb = self.shared_layer(x_emb)
            
        if self.D_projection == "spc":
            score = x_emb[:,0,:] # (N, d_modle)
        else:
            score = self.aggregation(x_emb) # (N, d_modle) or (N,L,d_model)
        
        if self.shared_layer is not None:
            score = self.discriminator_layers(score)
            
        score = self.regressor(score)
        return score
    
class CodePosterior(nn.Module): # bunch of encoder layers
    def __init__(self,dx:int, dc:int,
                 d_model:int, c_type:str, c_posterior_param:str,
                 shared_layer = None):
        super(CodePosterior, self).__init__()
        self.dx = dx
        self.d_model_c = d_model
        self.depth = 2
        self.dc = dc
        self.c_type = c_type
        self.c_posterior_param = c_posterior_param
        self.shared_layer = shared_layer
        
        if shared_layer is None:
            self.fC_projection = transformers.Conv1by1(self.dx, self.d_model_c)
            Q_layers =  [self.make_MLP() for _ in range(self.depth)]
            self.Q_layers = nn.ModuleList(Q_layers)
        else:
            self.Q_layers = nn.Sequential(src_utils.Linear(d_model, d_model),
                                                      nn.LeakyReLU(0.2),
                                                      nn.Linear(d_model, d_model),
                                                      nn.LeakyReLU(0.2)
                                                    #   ,
                                                    #   nn.LeakyReLU(0.2),
                                                    #   nn.Linear(d_model // 2, d_model)
                                                      )
        if c_type == "discrete":
            self.classifier = src_utils.Linear(self.d_model_c,self.dc)
            self.softmax = nn.Softmax(-1)
        elif c_type == "continuous":
            # soft fitting
            if c_posterior_param == "soft":
                # self.code_mu_logvar = src_utils.Linear(self.d_model_c, self.dc * 2)
                self.code_mu = src_utils.Linear(d_model, self.dc )
                self.code_logvar = src_utils.Linear(d_model, self.dc )
                # self.code_logvar.weight.data.fill_(-3)  # Initialize log-variance weights to prevent large sigma
                # self.code_logvar.bias.data.fill_(-3)
            # hard fitting
            else: self.code_mu = src_utils.Linear(self.d_model_c,self.dc)

    def forward(self, x):
        if self.shared_layer is None:
            code_emb = self.fC_projection(x)
            for layer in self.Q_layers:
                code_emb = layer(code_emb)
        else:
            code_emb = self.shared_layer(x)
            if self.shared_layer.D_projection == "spc":
                code_emb = code_emb[:,1:,:]
            code_emb = self.Q_layers(code_emb)
        if self.c_type == "discrete":
            logit = self.classifier(code_emb)
            return logit, None
        
        elif self.c_type == "continuous":
            if self.c_posterior_param == "soft":
                # code_emb = self.code_mu_logvar(code_emb)
                # mu, log_var = torch.chunk(code_emb, 2, dim=-1)
                mu = self.code_mu(code_emb)
                log_var = self.code_logvar(code_emb)
                # print("mu Q':", mu.mean())
                # print("logvar Q':", log_var.mean())
                mu = torch.clamp(mu, min=-1.5, max=1.5)
                log_var = torch.clamp(log_var, min=-4, max=0)
                return mu, log_var
            else:
                mu = self.code_mu(code_emb)
                mu = torch.clamp(mu, min=-1.5, max=1.5)
                # print("mu Q':", mu.mean())
                return mu, None
    def inference(self, x, logits = False):
        c, log_var = self.forward(x)
        # print("mu code':", c.mean())
        if log_var is not None:
            pass
            # print("logvar code':", log_var.mean())
        if self.c_type == "discrete":
            return torch.argmax(self.softmax(c),dim = -1, keepdim=True) if not logits else c#  
        else:
            return c
    def make_MLP(self):
        mlp = nn.Sequential(src_utils.Linear(self.d_model_c, self.d_model_c * 3),
                            nn.LeakyReLU(0.2),
                            src_utils.Linear(self.d_model_c * 3, self.d_model_c))
        return mlp 
    
