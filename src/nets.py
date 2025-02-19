import torch.nn as nn
import torch.nn.functional as F
import torch 

from src.modules import (encoders, generators, discriminator_Q, distributions, flow_transforms) 
from src.blocks import transformers


class InfoGAN(nn.Module):
    def __init__(self, 
                dx:int = 14, dz:int = 10, dc:int = 6, window:int = 25, d_model:int = 128, 
                num_heads:int = 4, z_projection:str = "aggregation", D_projection:str = "aggregation", 
                time_emb:bool = True, c_type:str = "discrete", c_posterior_param:str = "soft", encoder_E:str = "transformer",
                device = "cpu"): 
        super(InfoGAN, self).__init__()
        self.device = device
        self.c_type = c_type
        self.code_posterior_param = c_posterior_param
        # Prior Distributions p(z') and p(c)
        self.prior_z = distributions.DiagonalGaussian(dz, mean = 0, var = 1, device=device)
        # TODO there is distributional mismatch if use gasNLL or implement gaussian one for c 
        self.prior_c = distributions.UniformDistribution(device=device) if c_type == "continuous" \
                else distributions.DiscreteUniform(dc, onehot = True, device=device) 
                # else distributions.ContinuousCategorical(dc, 
                #                                         gumbel_temperature= 0.3,
                #                                         decay_rate= 0.95,
                #                                         dist="uniform",
                #                                         device=device)
                # else distributions.DiscreteUniform(dc, onehot = True, device=device) 
        self.h = flow_transforms.LatentFlow(dz, self.prior_z)
        
        self.G = generators.Decoder(dx=dx, dz=dz, dc=dc, window=window, d_model=d_model,
                                    num_heads=num_heads, p_h = self.h, p_c = self.prior_c)
        self.shared_net = transformers.shared_transformer(dx=dx, d_model=d_model, window=window, 
                                                    num_heads=num_heads, D_projection=D_projection)
        self.D = discriminator_Q.Discriminator(dx=dx, window=window, d_model=d_model,
                                               num_heads=num_heads, D_projection=D_projection,
                                               shared_layer= self.shared_net)
        self.Q = discriminator_Q.CodePosterior(dx=dx, dc=dc, d_model=d_model, c_type=c_type, 
                                               c_posterior_param=c_posterior_param,
                                               shared_layer=self.shared_net)
        
        
    def Loss_Discriminator(self, x):
        sample_size = x.shape[0]
        x_gen, _,_ = self.G.generation(sample_size)
        fake = self.D(x_gen)
        real = self.D(x)
        # print("D real:          ", real)
        # print("D fake:          ", fake)
        real_loss = torch.mean((real -1)**2)
        fake_loss = torch.mean((fake)**2) 
        return 0.5 * (real_loss + fake_loss), [real_loss, fake_loss] 
    def Loss_Generator(self, x, epoch = 1):
        if epoch is not None:
            annealing = min(.5, epoch / 20)
        else: annealing = 0.1
        
        sample_size = x.shape[0]
        x_gen, set_latent_samples, logdet = self.G.generation(sample_size)
        z, z0, c, c_logit = set_latent_samples

        gen = self.D(x_gen)
        q_code_mu, q_code_logvar = self.Q(x_gen)
            # - log_det
        # Generator loss
        gen_loss = 0.5 * torch.mean((gen - 1)**2)    
        if self.c_type == "continuous":
            if self.code_posterior_param == "soft":
                NLL_loss_Q = self.prior_c.NLL_gau(c, q_code_mu, q_code_logvar, "mean")
                
            elif self.code_posterior_param == "hard":
                NLL_loss_Q = self.prior_c.hard_fitting(c, q_code_mu)
            
        elif self.c_type == "discrete":
            # CE
            NLL_loss_Q = self.prior_c.cross_entropy_loss(c_logit if c_logit is not None else c, q_code_mu)
        
        
        # log_qz0 = self.prior_z.log_prob(z0)
        # log_qz = log_qz0 - logdet
        # # print("log_qz0: ", log_qz0.mean())
        # # print("logdet", logdet.mean())
        # log_pz = self.prior_z.log_prob(z)
        # # KL div & logdet
        # reverse_kl =  -(log_qz).mean() #- (log_pz).mean() # .mean()
        
        log_qz = self.prior_z.log_prob(z)
        reverse_kl = - (log_qz + logdet).mean()
        # reverse_kl *= annealing
        
        return gen_loss + NLL_loss_Q + (reverse_kl), [gen_loss, NLL_loss_Q, reverse_kl]
     
class VAE(nn.Module):
    def __init__(self, 
                dx:int = 14, dz:int = 10, dc:int = 6, window:int = 25, d_model:int = 128, 
                num_heads:int = 4, z_projection:str = "aggregation", D_projection:str = "aggregation", 
                time_emb:bool = True, c_type:str = "discrete", c_posterior_param:str = "soft", encoder_E:str = "transformer",
                device = "cpu"): 
        super(VAE, self).__init__()
        self.prior_z = distributions.DiagonalGaussian(dz, mean = 0, var = 1, device=device)
        # TODO there is distributional mismatch if use gasNLL or implement gaussian one for c 
        self.prior_c = None 
        # distributions.UniformDistribution(device=device) if c_type == "continuous" \
        #                         else distributions.DiscreteUniform(dc, onehot = True, device=device) 
        
        # NF transform
        self.h = flow_transforms.LatentFlow(dz, self.prior_z)
        
        self.f_E = encoders.LatentEncoder(dx=dx, dz=dz, window=window, d_model=d_model, 
                                    num_heads=num_heads, z_projection=z_projection, 
                                    time_emb=time_emb, encoder_E=encoder_E, p_h=self.h) 
        
        # self.f_C = encoders.CodeEncoder(dx=dx, dc=dc, d_model=d_model, 
        #                                 c_type=c_type, c_posterior_param=c_posterior_param)
        
        
        # Decoder
        self.f_D = generators.Decoder(dx=dx, dz=dz, dc=dc, window=window, d_model=d_model,
                                    num_heads=num_heads, p_h = self.h, p_c = self.prior_c)
        
    def Loss_VAE(self, x, tidx, epoch = None):
        if epoch is not None:
            annealing = min(1.0, epoch / 10)
        else: annealing = 1.0
        N, L, _ = x.shape
        # beta is a warm-up coefficient for preventing negative 
        # Encoding
        mu, log_var, zin = self.f_E(x, tidx)
        z, logdet, z0 = self.f_E.reparameterization_NF(mu, log_var)
        # c, c_logvar = self.f_C(x)
        # Decoding
        x_rec = self.f_D(z, c = None, zin = zin)
        
        # print("mu:  ", mu.mean())
        # print("log_var:  ", log_var.mean()) 
        # print("c", c)
        # print("c_logvar", c_logvar) 
        
        # Reconstruction & CC is implicitly made since G and f_D share the parameters
        # recon = F.mse_loss(x_rec, x, reduction = 'mean') # smooth_l1_loss 
        recon = F.mse_loss(x_rec, x, reduction = 'mean')# smooth_l1_loss 

        kl_div = (-0.5 * (1 + log_var - mu.pow(2) - log_var.exp()).sum(-1)).mean()
        
        return recon + (annealing*kl_div), [recon, annealing*kl_div] 

    def L_R(self, x, tidx, cond =None, epoch = None):
        if epoch is not None:
            annealing = min(0.1, epoch / 7)
        else: annealing = 0.01
        N, L, _ = x.shape
        # beta is a warm-up coefficient for preventing negative 
        # Encoding
        mu, log_var, zin = self.f_E(x, tidx)
        z, logdet, z0 = self.f_E.reparameterization_NF(mu, log_var)
        # print("mu!!:    ", mu[0:2,:])
        # print("log_var!!:    ", log_var[0:2,:])

        # print("z0!!:    ", z0[0:2,:])
        # print("z!!:    ", z[0:2,:])
        # c, c_logvar = self.f_C(x)
        # Decoding
        x_rec = self.f_D(z,c = cond, zin = zin)
        
        # print("mu:  ", mu.mean())
        # print("log_var:  ", log_var.mean()) 
        # print("c", c)
        # print("c_logvar", c_logvar) 
        
        # # Reconstruction & CC is implicitly made since G and f_D share the parameters
        recon = F.mse_loss(x_rec, x, reduction = 'sum') / (N+L)  # smooth_l1_loss 
        
        
        # kl_div = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp()).sum(-1)
        # kl_div -= logdet
        # kl_div = kl_div.mean()
        
        
        # print("z:  ",z)
        # print("z0:   ",z0)
        # logq(z'|x), where z' = z0  # Check this objective again
        
        log_qz0 = self.prior_z.log_prob(z0, mu, log_var)
        log_qz = log_qz0 - logdet
        # print("log_qz0: ", log_qz0.mean())
        # print("logdet", logdet.mean())
        log_pz = self.prior_z.log_prob(z)
        # KL div & logdet
        kl_div =  (log_qz - log_pz).mean() # .mean()
        
        # log_q_z0 = -0.5 * torch.sum(torch.log(2 * torch.tensor(torch.pi)) + log_var + (z0 - mu).pow(2) / log_var.exp(), dim=1)
        # log_p_zK = -0.5 * torch.sum(z0.pow(2), dim=1) # prior is standard normal. z_0 is the same size as z_K, as we use the change of variables formula.
        # print(logdet)
        # kl_div = torch.mean(log_q_z0 - log_p_zK - logdet)
        
        # print("logdet: ", logdet)
        # print(log_qz.mean())
        # print("logpz", log_pz.mean())
        # kl_div = torch.clamp(kl_div, min=0)
        
        # print("recon", recon)
        # print("kl_div", kl_div)
        
        # x 0.5 as the reconstruction is done through G (shared net) as well 
        return recon + (annealing*kl_div), [recon, annealing*kl_div]
    
class AE(nn.Module):
    def __init__(self, 
                dx:int = 14, dz:int = 10, dc:int = 6, window:int = 25, d_model:int = 128, 
                num_heads:int = 4, z_projection:str = "aggregation", D_projection:str = "aggregation", 
                time_emb:bool = True, c_type:str = "discrete", c_posterior_param:str = "soft", encoder_E:str = "transformer",
                device = "cpu"): 
        super(AE, self).__init__()
        self.prior_z = distributions.DiagonalGaussian(dz, mean = 0, var = 1, device=device)
        # TODO there is distributional mismatch if use gasNLL or implement gaussian one for c 
        self.prior_c = None 
        # distributions.UniformDistribution(device=device) if c_type == "continuous" \
        #                         else distributions.DiscreteUniform(dc, onehot = True, device=device) 
           
        self.f_E = encoders.LatentEncoder(dx=dx, dz=dz, window=window, d_model=d_model, 
                                    num_heads=num_heads, z_projection=z_projection, 
                                    time_emb=time_emb, encoder_E=encoder_E, p_h=None) 
        
        # self.f_C = encoders.CodeEncoder(dx=dx, dc=dc, d_model=d_model, 
        #                                 c_type=c_type, c_posterior_param=c_posterior_param)
        
        
        # Decoder
        self.f_D = generators.Decoder(dx=dx, dz=dz, dc=dc, window=window, d_model=d_model,
                                    num_heads=num_heads, p_h = None, p_c = None)
        
    def Loss_AE(self, x, tidx, cond = None, epoch = None):
        if epoch is not None:
            annealing = min(.1, epoch / 20)
        else: annealing = 0.1
        # beta is a warm-up coefficient for preventing negative 
        # Encoding
        mu, log_var, zin = self.f_E(x, tidx)
        # c, c_logvar = self.f_C(x)
        # Decoding
        x_rec = self.f_D(mu, c = cond, zin = zin)
        
        # print("mu:  ", mu.mean())
        # print("log_var:  ", log_var.mean()) 
        # print("c", c)
        # print("c_logvar", c_logvar) 
        
        # Reconstruction & CC is implicitly made since G and f_D share the parameters
        recon = F.mse_loss(x_rec, x, reduction = 'mean') # smooth_l1_loss 
        
        return recon, [recon] 