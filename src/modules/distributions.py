'''
Prior distributions, p(z') and p(c)
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class DiagonalGaussian(nn.Module):
    def __init__(self, dim, mean = 0, var = 1):
        super(DiagonalGaussian, self).__init__()
        self.dim = dim
        self.mean = mean
        self.var = var
        self.std = np.sqrt(var)  # Standard deviation (sqrt of variance)

    def sample(self, N, 
               mu = None, 
               log_var = None):
        # N shape (*,dz_dim)
        if isinstance(N, tuple):
            N_dim = N
        else: 
            N_dim = N.shape
        eps = torch.randn(N_dim)  # Standard normal random samples
        samples = eps + (mu if mu is not None else self.mean)
        samples *= torch.exp(0.5 * log_var) if log_var is not None else self.std
        return samples

    def log_prob(self, z, mu = None, log_var = None, normalize = True):
        """
        Computes the log probability of a given input x under the Gaussian distribution.
        """
        z_dim = z.shape[-1]
        if mu is not None: mu = mu
        else: mu = self.mean
        if log_var is not None: std = torch.exp(0.5 * log_var)
        else: std = self.std
            
        diff = z - mu
        log_prob = -0.5 * torch.sum(((diff / std) ** 2), dim=-1)  # sum over dimensions
        if normalize:
            log_prob -= 0.5 * z_dim * torch.log(2 * torch.pi)  # Normalizing constant
        log_prob -= torch.sum(torch.log(std))  # Log of the standard deviations (for each dim)
        return log_prob

    def H(self):
        """
        Computes the entropy of the diagonal Gaussian distribution.
        """
        # Entropy of a diagonal Gaussian is: H = 0.5 * (dim * log(2 * pi * e) + sum(log(var)))
        entropy = 0.5 * (self.dim * torch.log(2 * torch.pi * torch.exp(torch.tensor(1.0))) + torch.sum(torch.log(self.var)))
        return entropy
    
    def NLL(self, z, mu, log_var, aggr = "mean"):
        return -torch.sum(self.log_prob(z, mu, log_var)) if aggr == "sum" else -torch.mean(self.log_prob(z, mu, log_var))

    def hard_fitting(self, z_true, z_mu):
        return F.l1_loss(z_mu, z_true, reduction = 'mean')
    
class UniformDistribution(nn.Module):
    def __init__(self, low=-1., high=1.0):
        """
        Uniform distribution
        """
        super(UniformDistribution, self).__init__()
        self.low = low
        self.high = high

    def sample(self, N, target:float = None):
        # N shape (*,dc_dim)
        if isinstance(N, tuple):
            N_dim = N
        else: 
            N_dim = N.shape
        if target is None:
            return torch.rand(N_dim) * (self.high - self.low) + self.low
        else:
            if (target > self.high) or (target < self.low):
                target = 0.1
            return torch.ones(N_dim) * target
        
    def log_prob(self, c):
        """
        Computes the log probability of a given input c under the uniform distribution.
        """
        # For uniform distribution, log_prob is -log(high - low) if z is within the bounds.
        log_prob = torch.full_like(c, -torch.log(torch.tensor(self.high - self.low)))
        # Check if z is within bounds [low, high]
        log_prob = torch.where((c >= self.low) & (c <= self.high), log_prob, torch.tensor(-float('inf')))
        return log_prob

    def log_prob_gaussian(self, c, mu = None, log_var = None, normalize = True):
        """
        This is used for Maximizing mutual information in training Q net for soft fitting 
        Note that this log prob is over gaussian not uniform!
        
        """
        c_dim = c.shape[-1]
        if mu is not None: mu = mu
        else: mu = self.mean
        if log_var is not None: std = torch.exp(0.5 * log_var)
        else: std = self.std
            
        diff = c - mu
        log_prob = -0.5 * torch.sum(((diff / std) ** 2), dim=-1)  # sum over dimensions
        if normalize:
            log_prob -= 0.5 * c_dim * torch.log(2 * torch.pi)  # Normalizing constant
        log_prob -= torch.sum(torch.log(std))  # Log of the standard deviations (for each dim)
        return log_prob
    
    def H(self):
        """
        Computes the entropy of the uniform distribution.
        """
        return torch.log(torch.tensor(self.high - self.low))
    
    def NLL(self, c, aggr = "sum"):
        return -torch.sum(self.log_prob(c)) if aggr == "sum" else -torch.mean(self.log_prob(c))
    
    def NLL_gau(self, c, mu, log_var, aggr = "mean"):
        return -torch.sum(self.log_prob_gaussian(c, mu, log_var)) if aggr == "sum" else -torch.mean(self.log_prob_gaussian(c, mu, log_var))
    
    def hard_fitting(self, c_true, c_mu):
        return F.l1_loss(c_mu, c_true, reduction = 'mean')
    
class DiscreteUniform(nn.Module):
    def __init__(self, num_classes=10, onehot = False):
        """
        Discrete uniform. 
        """
        super(DiscreteUniform, self).__init__()
        self.onehot = onehot
        self.low = 0
        self.num_classes = num_classes  # Number of discrete values

    def sample(self, N, target = None):
        # N has shape of (*, dc_dim)
        if isinstance(N, tuple):
            N_dim = N[:-1]
        else: 
            N_dim = N.shape[:-1]

        if target is not None:
            if (target > self.num_classes-1) or (target < 0):
                target = 1
            c = torch.ones(N_dim + (1,), dtype = int) * target
        else:
            c = torch.randint(self.low, self.num_classes, N_dim + (1,)) # (*,1)
        
        if self.onehot:
            c = torch.flatten(c, 0, -1)
            c = F.one_hot(c,num_classes = self.num_classes).view(N_dim+(self.num_classes,)) # (*, num_classes)
            print('!')
        else: pass
        return c

    def log_prob(self, z):
        """
        Computes the log probability of a given input z under the discrete uniform distribution.
        """
        # Ensure that z values are in the valid range [low, high]
        assert torch.all((z >= self.low) & (z <= self.num_classes -1)), "Sample values are out of range."

        # Each value in the range [low, high] has equal probability
        log_prob = -torch.log(torch.tensor(self.num_classes, dtype=torch.float32))  # log(1 / num_classes)
        return log_prob

    def cross_entropy_loss(self, y, logits): # TODO check later on 
        if self.onehot:
            # We apply softmax to the logits to convert them into probabilities
            probs = torch.softmax(logits, dim=-1)
            loss = -torch.sum(y * torch.log(probs), dim=-1)
            return torch.mean(loss)
        else:
            tiny = 1e-7  # for numerical stability
            N = logits.shape[0]
            log_probs = torch.log(logits + tiny)
            
            # Use the integer labels to index the appropriate log probabilities
            log_prob_for_y = log_probs[range(N), y]  # Get log probabilities corresponding to the true classes
            return -torch.mean(log_prob_for_y)  # Negative log likelihood loss (mean over batch)