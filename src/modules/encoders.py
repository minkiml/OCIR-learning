'''
Encoder f_E

Encoder f_C

'''
import torch.nn.functional as F
import torch
import torch.nn as nn

from src.blocks import transformers, src_utils 
from src.modules import distributions

class LatentEncoder(nn.Module):
    def __init__(self, args,
                 p_h
                 ): 
        super(LatentEncoder, self).__init__() 
        
        self.z_projection = args.z_projection
        self.num_heads = args.num_heads
        self.depth = 2
        self.time_emb = args.time_embedding
        if args.encoder_E == "transformer":
            # Suppose the longest length of cm data could be 520
            if self.time_emb:
                self.time_embedding = nn.Embedding(num_embeddings= args.window, embedding_dim = args.d_model)
                src_utils.init_embedding(self.time_embedding)
            self.pos_enc = transformers.SinCosPositionalEncoding(args.d_model, args.window + 2)
            self.fE_projection = transformers.Conv1by1(args.dx, args.d_model)
            
            # (N, L, C) -> (N, C)
            if args.z_projection == "aggregation":
                self.latent_aggregation = transformers.Aggregation(args.d_model, method = 'weighted')
                
            elif args.z_projection == "spc":
                # otherwise, a BERT-style special token (namely compressive token) is done 
                self.compressive_token = nn.Parameter(torch.randn(1,1,args.d_model) * 0.02)
                
            elif args.z_projection == "rnn":
                self.latent_aggregation = src_utils.rnn_aggregation(args.d_model, args.d_model)


            TransformerEncoder = [transformers.TransformerEncoderBlock(embed_dim = args.d_model, num_heads = self.num_heads,
                                                                       ff_hidden_dim = int(args.d_model * 3), dropout = 0.15,
                                                                       prenorm = True) for _ in range(self.depth)]
            self.TransformerEncoder = nn.ModuleList(TransformerEncoder)
        
        # mu and logvar
        self.mu_logvar = src_utils.Linear(args.d_model, args.dz * 2)
        # z' ~ N(mu, sigma) -> p_h(z)
        self.p_h = p_h
    def forward(self, x, tidx = None):
        N, L, c = x.shape
        x_proj = self.fE_projection(x)
        
        # Time index embeddings
        if self.time_emb:
            if tidx is not None: 
                time_token = self.time_embedding(tidx) # (N, 1, d_model)
            else:
                # For generator-to-encoder inference the time index is not known, so time index 0 is assigned as a unknown time. 
                time_token = self.time_embedding(torch.zeros((N,1), dtype=torch.long).to(x.device))
            x_emb = torch.cat((x_proj,time_token), dim = 1)
        else: x_emb = x_proj
        # add apc as a compressive token
        if self.z_projection == "spc":
            compressive_token = self.compressive_token.expand(N,-1,-1)
            x_emb = torch.cat((compressive_token, x_proj), dim = 1)            
            
        # positional encoding
        x_emb = self.pos_enc(x_emb)
        for layer in self.TransformerEncoder:
            x_emb = layer(x_emb)
        
        if self.z_projection == "spc":
            z = x_emb[:,0,:] # (N, d_modle)
        else:
            z = self.latent_aggregation(x_emb) # (N, d_modle)
        mu_logvar = self.mu_logvar(z)
        mu, log_var = torch.chunk(mu_logvar, 2, dim=-1)
        return mu, log_var
    
    def reparameterization_NF(self, mu, log_var):
        # reparameterization trick with flow transform
        z, logdet, z0 = self.p_h(log_var, mu, log_var)
        return z, logdet, z0
    
    def encoding(self, x, tidx):
        mu, logvar = self.forward(x, tidx)
        z, _, _ = self.reparameterization_NF(mu, logvar)
        return z
    
class CodeEncoder(nn.Module):
    def __init__(self, args): 
        super(CodeEncoder, self).__init__() 
        self.dx = args.dx
        self.d_model_c = args.d_model
        self.depth = 2
        self.dc = args.dc
        self.c_type = args.c_type
        self.c_posterior_param = args.c_posterior_param
        
        mlp_encoder = [self.make_MLP() for _ in range(self.depth)]
        self.mlp_encoder = nn.ModuleList(mlp_encoder)
            
        self.fC_projection = transformers.Conv1by1(self.dx, self.d_model_c)
        
        if args.c_type == "discrete":
            self.classifier = src_utils.Linear(self.d_model_c,self.dc)
            self.softmax = nn.Softmax(-1)
        elif args.c_type == "continuous":
            # soft fitting
            if args.c_posterior_param == "soft":
                self.code_mu_logvar = src_utils.Linear(self.d_model_c, self.dc * 2)
            # hard fitting
            else: 
                self.code_mu = src_utils.Linear(self.d_model_c,self.dc)

    def forward(self, x):
        code_emb = self.fC_projection(x)
        N, L, c = x.shape
        x = x.view(-1, c)
        
        for layer in self.mlp_encoder:
            code_emb = layer(code_emb)
            
        if self.c_type == "discrete":
            logit = self.classifier(code_emb)
            return logit.view(N,L,self.dc), None
        elif self.c_type == "continuous":
            if self.c_posterior_param == "soft":
                code_emb = self.code_mu_logvar(code_emb)
                mu, log_var = torch.chunk(code_emb, 2, dim=-1)
                return mu.view(N,L,self.dc), log_var.view(N,L,self.dc)
            else:
                mu = self.code_mu(code_emb)
                return mu.view(N,L,self.dc), None
    
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