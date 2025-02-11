'''
Decoder, f_D

Generator, G

Note that the decoder and generator in OCIR shares the same parameters
'''

import torch.nn.functional as F
import torch
import torch.nn as nn

from src.blocks import transformers, src_utils



class Decoder(nn.Module):
    def __init__(self, args,
                 
                 p_h, p_c
                 ): 
        super(Decoder, self).__init__() 
        self.depth = 2
        self.window = args.window
        self.dc = args.dc
        self.dz = args.dz
        self.num_heads = args.num_heads
        
        self.latent_decoder = src_utils.rnn_decoding_seqtoken(args.dz, args.dc,
                                                              hidden_dim= args.d_model,
                                                              window = args.window)
        
        TransformerDecoder = [transformers.TransformerEncoderBlock(embed_dim = args.d_model, num_heads = self.num_heads,
                                                                       ff_hidden_dim = int(args.d_model * 3), dropout = 0.15,
                                                                       prenorm = True) for _ in range(self.depth)]
        self.TransformerDecoder = nn.ModuleList(TransformerDecoder)
        self.final_layer = nn.Linear(args.d_model, args.dx)
        
        # p_h(z) & p(c)
        self.h = p_h
        self.p_c = p_c
    def forward(self, z, c):
        # z (N, dz)
        # c (N, L, dc)
        
        z_tokens = self.latent_decoder(z, c)
        
        for layer in self.TransformerDecoder:
            z_tokens = layer(z_tokens)
        x = self.final_layer(z_tokens)
    
        return x 
    
    def generation(self, num_samples, fixed_code = None):
        # N is the target shape in tuple
        
        N_c = (num_samples, self.window, self.dc)
        N_z = (num_samples, self.dz)
        # sample z ~ p(z') -> p_h(z)
        z, z0 = self.h.sample(N_z) # (N, L, dz)
        
        # sample c ~ p(c)
        c = self.p_c.sample(N_c, fixed_code) # (N, L, dc)
        print("c in generator", c)
        x = self.forward(z, c)
        return x, [z,z0, c]
    
    
        