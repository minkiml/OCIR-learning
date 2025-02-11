'''
Normalizing flow - based transformation, h 
'''

import torch
import torch.nn as nn

from src.blocks import normalizing_flow

# TODO:
class LatentFlow(nn.Module):
    '''
    Flow-based transform 
    
    It is based on the following convention
    
    forward h() : z' -> z
    
    backward h-1() : z -> z'
    '''
    def __init__(self, args, 
                 base_distribution = None):
        super(LatentFlow, self).__init__() 
        # p(z')
        self.base_distribution = base_distribution
        # h0,...,hL 
        self.normalizing_flow = normalizing_flow.NormalizingFlow("RNVP", dz = args.dz, 
                                                                 num_layers= 6)
        
    def forward(self, N, mu  = None, log_var = None, z0 = None):
        if z0 == None:
            z_0 = self.base_distribution.sample(N, mu, log_var)
        else: z_0 = z0
        z, log_det = self.normalizing_flow(z_0) # (N, dz)  , (N, ) 
        return z, log_det, z_0
    
    def inverse(self, x):
        # 
        z0, log_det = self.normalizing_flow.inverse(x) # (N, dz)  , (N, )
        return z0, log_det 
        
    def sample(self, N):
        z, _, z0= self.forward(N, mu = None, log_var = None)
        return z, z0
        