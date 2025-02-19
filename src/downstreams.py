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
    
class LatentTrajectory(nn.Module):
    def __init__(self, args, 
                    pretrained_encoderE,
                 device): 
        super(LatentTrajectory, self).__init__()
        
        
        
class DataTrajectory(nn.Module):
    def __init__(self, args, 
                    pretrained_encoderE,
                 device): 
        super(DataTrajectory, self).__init__()