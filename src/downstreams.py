import torch.nn as nn
from src.blocks import src_utils
from src.modules import encoders


class RulEstimator(nn.Module):
    def __init__(self, args, 
                    pretrained_encoderE:encoders.LatentEncoder,
                    device): 
        super(RulEstimator, self).__init__()
        self.device = device
        
        self.encoder = pretrained_encoderE    
        self.regressor = src_utils.Linear(args.dz, 1, bias = False)
        
    def forward(self, x, tidx = None):
        z = self.encoder.encoding(x, tidx)
        rul = self.regressor(z)
        return rul
    
class latent_trajectory(nn.Module):
    def __init__(self, args, 
                    pretrained_encoderE,
                 device): 
        super(latent_trajectory, self).__init__()
        
        
        
class data_trajectory(nn.Module):
    def __init__(self, args, 
                    pretrained_encoderE,
                 device): 
        super(data_trajectory, self).__init__()