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
    def __init__(self, args):
        super(Discriminator, self).__init__()

        self.depth = 2 
        self.num_heads = args.num_heads
        self.D_projection = args.D_projection
        # nn.LeakyReLU(0.2),
        self.pos_enc = transformers.SinCosPositionalEncoding(args.d_model, args.window + 1)
        self.fE_projection = transformers.Conv1by1(args.dx, args.d_model)
        TransformerEncoder = [transformers.TransformerEncoderBlock(embed_dim = args.d_model, num_heads = self.num_heads,
                                                                       ff_hidden_dim = int(args.d_model * 3), dropout = 0.15,
                                                                       prenorm = True) for _ in range(self.depth)]
        self.TransformerEncoder = nn.ModuleList(TransformerEncoder)
        
        # (N, L, C) -> (N, C)
        if args.D_projection == "aggregation": 
            self.aggregation = transformers.Aggregation(args.d_model, method = 'weighted')
            
        elif args.D_projection == "spc":
            # otherwise, a BERT-style special token (namely compressive token) is done 
            self.score_token = nn.Parameter(torch.randn(1,1,args.d_model) * 0.02)
            
        elif args.D_projection == "rnn":
            self.aggregation = src_utils.rnn_aggregation(args.d_model, args.d_model)
        
        elif args.D_projection == "None":
            self.aggregation = nn.Identity() 
        
        self.regressor = src_utils.Linear(args.d_model, 1)
        
    def forward(self, x):
        # x (N, L, c)
        N, L, c = x.shape
        x_emb = self.fE_projection(x)
        if self.D_projection == "spc":
            score_token = self.score_token.expand(N,-1,-1)
            x_emb = torch.cat((score_token, x_emb), dim = 1)                   
        # positional encoding
        x_emb = self.pos_enc(x_emb)
        for layer in self.TransformerEncoder:
            x_emb = layer(x_emb)
        
        if self.D_projection == "spc":
            score = x_emb[:,0,:] # (N, d_modle)
        else:
            score = self.aggregation(x_emb) # (N, d_modle) or (N,L,d_model)
        score = self.regressor(score)
        return score
    
class CodePosterior(nn.Module): # bunch of encoder layers
    def __init__(self,args):
        super(CodePosterior, self).__init__()
        self.dx = args.dx
        self.d_model_c = args.d_model
        self.depth = 2
        self.dc = args.dc
        self.c_type = args.c_type
        self.c_posterior_param = args.c_posterior_param
        
        self.fC_projection = transformers.Conv1by1(self.dx, self.d_model_c)
        
        mlp_encoder =  [self.make_MLP() for _ in range(self.depth)]
        self.mlp_encoder = nn.ModuleList(mlp_encoder)
        
        if args.c_type == "discrete":
            self.classifier = src_utils.Linear(self.d_model_c,self.dc)
            self.softmax = nn.Softmax(-1)
        elif args.c_type == "continuous":
            # soft fitting
            if args.c_posterior_param == "soft":
                self.code_mu_logvar = src_utils.Linear(self.d_model_c, self.dc * 2)

            # hard fitting
            else: self.code_mu = src_utils.Linear(self.d_model_c,self.dc)

    def forward(self, x):
        code_emb = self.fC_projection(x)
        N, L, c = x.shape
        x = x.view(-1, c)
        
        for layer in self.mlp_encoder:
            code_emb = layer(code_emb)
        
        if self.c_type == "discrete":
            logit = self.classifier(code_emb)
            return logit, None
        
        elif self.c_type == "continuous":
            if self.c_posterior_param == "soft":
                code_emb = self.code_mu_logvar(code_emb)
                mu, log_var = torch.chunk(code_emb, 2, dim=-1)
                return mu, log_var
            else:
                mu = self.code_mu(code_emb)
                return mu, None
    def inference(self, x):
        c, _ = self.forward(x)
        if self.c_type == "discrete":
            return torch.argmax(self.softmax(c),dim = -1, keepdim=True) #  
        else:
            return c
    def make_MLP(self):
        mlp = nn.Sequential(src_utils.Linear(self.d_model_c, self.d_model_c * 3),
                            nn.LeakyReLU(0.2),
                            src_utils.Linear(self.d_model_c * 3, self.d_model_c))
        return mlp 
    
