import torch
import torch.nn as nn
# TODO shape checking

class rnn_aggregation(nn.Module):
    def __init__(self,input_dim = 10, hidden_dim = 32, 
                 dropout_ = 0., layer = 1):
        super(rnn_aggregation, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer = layer
        self.gru = nn.GRU(input_dim, hidden_dim,
                                  num_layers= layer,
                                  batch_first = True,
                                  bias = True,
                                  dropout = dropout_)
    def forward(self, x):
        h0 = torch.zeros((self.layer, x.shape[0], self.hidden_dim)).to(x.device)
        _, h_t = self.gru(x,h0)
        return h_t[-1,:,:] # (batch, hidden)
    
class rnn_decoding_eq(nn.Module):
    def __init__(self,dz, dc, 
                 hidden_dim = 32, 
                 dropout_ = 0., layer = 1, 
                 noise = True):
        super(rnn_decoding_eq, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer = layer
        self.noise = noise
        self.gru = nn.GRU(dz + dc, 
                          hidden_dim,
                            num_layers= layer,
                            batch_first = True,
                            bias = True,
                            dropout = dropout_)
        self.linear_layer = nn.Linear(hidden_dim, hidden_dim, bias = False)
    def forward(self, z, c):
        N, L, dc = c.shape
        _,dz = z.shape
        z = z.unsqueeze(1).repeat(1, L, 1) # expansion  or expand(N, L, dz)
        if self.noise:
            z += torch.randn(N, L, dz).to(z.device) * 0.05
        z_c = torch.concatenate((z,c), dim = -1)
        
        h0 = torch.zeros((self.layer, z_c.shape[0], self.hidden_dim)).to(z_c.device)
        z_c, _ = self.gru(z_c, h0)
        
        return self.linear_layer(z_c) # (batch, L, hidden)
    
class rnn_decoding_seqtoken(nn.Module):
    '''Use the input z as starting state to propagate'''
    def __init__(self, dz, dc, hidden_dim = 32, 
                 dropout_ = 0., layer = 1, 
                 learnable = True, window = 100):
        super(rnn_decoding_seqtoken, self).__init__()# TODO 
        self.hidden_dim = hidden_dim
        self.dz = dz
        self.layer = layer
        self.gru = nn.GRU(dz + dc, hidden_dim,
                                  num_layers= layer,
                                  batch_first = True,
                                  bias = True,
                                  dropout = dropout_)
        if learnable:
            self.seqtoken = nn.Parameter(torch.randn(1,window, dz) * 0.02)
        else:
            seqtoken = torch.randn(1, window, dz) * 0.02
            self.register_buffer("seqtoken", seqtoken)
        self.expansion = nn.Linear(dz,hidden_dim, bias = False)
    def forward(self, z, c):
        N, L, dc = c.shape
        _, dz = z.shape
        seqtoken = self.seqtoken.expand(N, L, self.dz).to(z.device) # expansion
        
        c = torch.flip(c, dims = [1]) # flip the seq indices so that the hidden state can influence from the last 
        seqtoken_c = torch.concatenate((seqtoken,c), dim = -1)
        
        h0 = self.expansion(z)
        h0 = h0.unsqueeze(0).repeat(self.layer, 1, 1)
        z_c, _ = self.gru(seqtoken_c, h0)
        z_c = torch.flip(z_c, dims = [1]) # flip back to the original order
        return z_c

class LayerNorm(nn.Module):
    def __init__(self, num_channels, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, num_channels, 1))  
        self.beta = nn.Parameter(torch.zeros(1, num_channels, 1))  

    def forward(self, x):
        # x: (B, C, L)
        mean = x.mean(dim=1, keepdim=True)  
        var = x.var(dim=1, unbiased=False, keepdim=True)  

        x_norm = (x - mean) / torch.sqrt(var + self.eps) 
        return self.gamma * x_norm + self.beta  
    
    
class Sine(nn.Module):
    def __init__(self):
        super(Sine, self).__init__()
    def forward(self, input):
        return torch.sin(input)