'''
Decoder, f_D

Generator, G

Note that the decoder and generator in OCIR shares the same parameters
'''

import torch.nn.functional as F
import torch.nn as nn

import src.blocks as bs
from src.modules import distributions

class Decoder(nn.Module):
    def __init__(self, dx:int, dz:int, dc:int, window:int, 
                 d_model:int, num_heads:int, p_h, p_c = None
                ): 
        super(Decoder, self).__init__() 
        self.depth = 2
        self.window = window
        self.dc = dc #if p_c is not None else 0
        self.dz = dz
        self.num_heads = num_heads

        # self.latent_decoder = bs.rnn_decoding_seqtoken(dz, self.dc,
        #                                                       hidden_dim= d_model,
        #                                                       window = window,
        #                                                       seq_out=False)
        self.latent_decoder = bs.wide_decoding(dz, self.dc, hidden_dim= d_model,
                                               window = window)
        # self.latent_decoder = bs.comb_decoding(dz, self.dc, hidden_dim= d_model,
        #                                        window = window)
        
        # self.latent_decoder = bs.rnn_decoding(dz, self.dc,
        #                                                       hidden_dim= d_model,
        #                                                       window = window)
        # self.latent_decoder = bs.rnn_decoding_eq(dz, self.dc, 
        #                                                 hidden_dim = d_model, 
        #                                                 window = window)
      
        # self.latent_decoder = bs.decoding_tokens(dz, self.dc,
        #                                         hidden_dim= d_model,
        #                                         window = window)
        
        TransformerDecoder = [bs.TransformerEncoderBlock(embed_dim = d_model, num_heads = self.num_heads,
                                                                       ff_hidden_dim = int(d_model * 3), dropout = 0.15,
                                                                       prenorm = True) for _ in range(self.depth)]
        self.TransformerDecoder = nn.ModuleList(TransformerDecoder)
        
        # TransformerDecoder = [bs.TCN_net(max_input_length = window, # This determins the maximum capacity of sequence length
        #                             input_size = d_model,
        #                             kernel_size = 3,
        #                             num_filters = d_model,
        #                             num_layers = None,
        #                             dilation_base = 2,
        #                             norm= 'weightnorm', # "none1" 
        #                             nr_params = 1,
        #                             dropout= 0.1) for _ in range(self.depth)]
        # self.TransformerDecoder = nn.ModuleList(TransformerDecoder)


        
        self.final_layer = bs.Linear(d_model, dx, bias=False)
        
        # self.layernorm = nn.LayerNorm(dz)
        # p_h(z) & p(c)
        self.h = p_h
        self.p_c = p_c
    def forward(self, z, c = None, generation = False, zin = None):
        # z (N, dz)
        # c (N, L, dc)
        if not generation: 
            # At Inference where the arg c is logit inferred by f_C
            if isinstance(self.p_c, distributions.DiscreteUniform):
                c = F.softmax(c,dim = -1).detach()
            elif isinstance(self.p_c, distributions.ContinuousCategorical):
                c = self.p_c.gumbel_softmax(c)
            elif isinstance(self.p_c, distributions.UniformDistribution):
                c = c
            else:
                c = c
            # elif self.p_c == None:
            #     c = None
        # z = self.layernorm(z)
        z_tokens = self.latent_decoder(z, c, zin)
        
        for layer in self.TransformerDecoder:
            z_tokens = layer(z_tokens)
        x = self.final_layer(z_tokens)
        return x 
    
    def generation(self, num_samples, fixed_code = None):
        # N is the target shape in tuple
        
        N_c = (num_samples, self.window, self.dc)
        N_z = (num_samples, self.dz)
        # sample z ~ p(z') -> p_h(z)
        if isinstance(self.h, distributions.DiagonalGaussian):
            z = self.h.sample(N_z) # (N, L, dz)
            z0 = None
        else:
            z, z0, logdet = self.h.sample(N_z) # (N, L, dz)
        
        # sample c ~ p(c)
        if self.p_c is not None:
            c = self.p_c.sample(N_c, fixed_code) # (N, L, dc)
            if self.p_c.gumbel:
                c, c_logit = c
            else:
                c_logit = None
        else:
            c, c_logit = None, None            
        x = self.forward(z, c, generation = True)
        return x, [z,z0, c, c_logit], logdet
    
    
        