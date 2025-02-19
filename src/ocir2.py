import torch.nn as nn
import torch.nn.functional as F
import torch 

from src.modules import (encoders, generators, discriminator_Q, distributions, flow_transforms) 
from src.blocks import transformers

'''TODO: 
1) visualization, metrics

    check dataloading (& full framework once on single minibatch) 

    rul estimator and task 

    ** check data loading first then! 
    ** check full framework
    ** implement visualization
    ** Think about and Implemt rul estimator 

2) loading pipeline 

    trajectory model and task




'''
class OCIR2(nn.Module):
    def __init__(self, 
                dx:int = 14, dz:int = 10, dc:int = 6, window:int = 25, d_model:int = 128, 
                num_heads:int = 4, z_projection:str = "aggregation", D_projection:str = "aggregation", 
                time_emb:bool = True, c_type:str = "discrete", c_posterior_param:str = "soft", encoder_E:str = "transformer",
                device = "cpu"): 
        super(OCIR2, self).__init__()
        self.device = device
        self.c_type = c_type
        self.z_projection = z_projection
        self.time_emb = time_emb
        self.code_posterior_param = c_posterior_param
        
        # Prior Distributions p(z') and p(c)
        self.prior_z = distributions.DiagonalGaussian(dz, mean = 0, var = 1, device=device)
        # TODO there is distributional mismatch if use gasNLL or implement gaussian one for c 
        self.prior_c = distributions.UniformDistribution(device=device) if c_type == "continuous" \
           else distributions.DiscreteUniform(dc, onehot = True, device=device) 
                # else distributions.ContinuousCategorical(dc, 
                #                                         gumbel_temperature= 0.4,
                #                                         decay_rate= 0.95,
                #                                         dist="uniform",
                #                                         device=device)
                # else distributions.DiscreteUniform(dc, onehot = True, device=device) 

        # NF transform
        self.h = flow_transforms.LatentFlow(dz, self.prior_z)
        
        # shared early layers betweew f_E and f_C
        self.shared_encoder_layers = None
        # transformers.SharedEncoder(dx=dx, dz=dz, window=window, d_model=d_model, 
        #                                   num_heads=num_heads,z_projection=z_projection, time_emb=time_emb)
        # Encoders
        self.f_E = encoders.LatentEncoder(dx=dx, dz=dz, window=window, d_model=d_model, 
                                          num_heads=num_heads, z_projection=z_projection, 
                                          time_emb=time_emb, encoder_E=encoder_E, p_h=self.prior_z, shared_EC= True if self.shared_encoder_layers is not None else False) 
        
        self.f_C = encoders.CodeEncoder(dx=dx, dc=dc, d_model=d_model, 
                                        c_type=c_type, c_posterior_param=c_posterior_param, shared_EC= True if self.shared_encoder_layers is not None else False)
        
        # Generator and decoder
        self.G = generators.Decoder(dx=dx, dz=dz, dc=dc, window=window, d_model=d_model,
                                    num_heads=num_heads, p_h = self.h, p_c = self.prior_c)
        self.f_D =  generators.Decoder(dx=dx, dz=dz, dc=dc, window=window, d_model=d_model,
                                    num_heads=num_heads, p_h = self.h, p_c = self.prior_c) #self.G # shared
        
        # Discriminator and Q
        self.shared_net = transformers.shared_transformer(dx=dx, d_model=d_model, window=window, 
                                                          num_heads=num_heads, D_projection=D_projection)
        # self.shared_net = None
        self.D = discriminator_Q.Discriminator(dx=dx, window=window, d_model=d_model,
                                               num_heads=num_heads, D_projection=D_projection,
                                               shared_layer= self.shared_net)
        self.Q = discriminator_Q.CodePosterior(dx=dx, dc=dc, d_model=d_model, c_type=c_type, 
                                               c_posterior_param=c_posterior_param,
                                               shared_layer=self.shared_net)
    
    
    '''
    For our objective function, L = L_R + L_G + L_RG
    it can be doen in two steps 
    
    1) L_R & L_RG(x -> E -> G)
    Since decoder and G share the same parameters, the objective can be further simplified to L_NF + L_VAE
    L_R = logq(x|z) - KL(logq_h(z|x), logp(z) ) 
    
    2) L_G & L_RG ({z,c} -> G -> E)
    where we embedd the objective of L_RG in L_G.
    '''
    
    # Objective functions
    def L_R(self, x, tidx, epoch = None):
        if epoch is not None:
            annealing = min(1.0, epoch / 7)
        else: annealing = 1.0
        # beta is a warm-up coefficient for preventing negative 
        # Encoding
        if self.shared_encoder_layers is not None:
            h = self.shared_encoder_layers(x, tidx)
            
            hc = h
            if (self.z_projection == "spc") or (self.z_projection == "seq"):
                hc = hc[:,1:,:]
            if self.time_emb:
                hc = hc[:,:-1,:]
        else: 
            h = x
            hc = x
        mu, log_var, zin = self.f_E(h, tidx)
        z, _, _ = self.f_E.reparameterization_NF(mu, log_var)
        
        c, _ = self.f_C(hc)
        # Decoding
        x_rec = self.f_D(z,c, zin = zin)
        x_rec_G = self.G(z.detach(),c.detach(), zin = zin)
        
        # Reconstruction & CC is implicitly made since G and f_D share the parameters
        recon = F.mse_loss(x_rec, x, reduction = 'mean') 
        recon_G = F.mse_loss(x_rec_G, x, reduction = 'mean') 
        recon_G = recon_G * 0.05
        
        # KL in ELBO
        kl_div = (-0.5 * (1 + log_var - mu.pow(2) - log_var.exp()).sum(-1)).mean()
        kl_div = annealing * kl_div
        # Forward KL
        z0, logdet,_ = self.h.inverse(mu.detach())
        MLE_loss = (-(self.prior_z.log_prob(z0)) + logdet).mean() # by NLL   might be - logdet

        return recon + (MLE_loss) + kl_div + recon_G, [recon, kl_div, MLE_loss, recon_G]
    
    def L_G_discriminator(self, x):
        sample_size = x.shape[0]
        x_gen, _,_ = self.G.generation(sample_size)
        fake = self.D(x_gen)
        real = self.D(x)
        # print("D real:          ", real)
        # print("D fake:          ", fake)
        real_loss = torch.mean((real -1)**2)
        fake_loss = torch.mean((fake)**2)
        
        # print("real_loss", real_loss)
        # print("fake_loss", fake_loss)

        return 0.5 * (real_loss + fake_loss), [real_loss, fake_loss]
    
    def L_G_generator(self, x):
        sample_size = x.shape[0]
        x_gen, set_latent_samples, log_det = self.G.generation(sample_size)
        z, z0, c, c_logit = set_latent_samples #TODO yield logit samples when gumbel is used
        
        gen = self.D(x_gen)
        # print("gen:          ", gen)
        q_code_mu, q_code_logvar = self.Q(x_gen)
        
        # CC
        if self.shared_encoder_layers is not None:
            h = self.shared_encoder_layers(x_gen) # .detach()
            hc = h
            if (self.z_projection == "spc") or (self.z_projection == "seq"):
                hc = hc[:,1:,:]
            if self.time_emb:
                hc = hc[:,:-1,:]
        else: 
            h = x_gen#.detach()
            hc = x_gen#.detach()
            
        mu, _, _ = self.f_E(h)
        c_gen, _ = self.f_C(hc)


        # Generator loss
        gen_loss = 0.5 * torch.mean((gen - 1)**2)
        
        # print("x_gen", x_gen)
        # print("q_code_mu", q_code_mu)
        # print("q_code_logvar", q_code_logvar) 
        
        # CC and MMI for Q, G  
        # cc_loss_z = self.prior_z.NLL(z0, mu, logvar, "mean") # soft fitting
        cc_loss_z = self.prior_z.hard_fitting(z.detach(), mu) # z0 & self.h.inverse(mu) or z & mu
        
        if self.c_type == "continuous":
            if self.code_posterior_param == "soft":
                # NLL
                # cc_loss_c = self.prior_c.NLL_gau(c, c_gen, c_logvar, "sum")
                NLL_loss_Q = self.prior_c.NLL_gau(c, q_code_mu, q_code_logvar, "mean")
                
            elif self.code_posterior_param == "hard":
                # MSE
                # cc_loss_c = self.prior_c.hard_fitting(c, c_gen)
                NLL_loss_Q = self.prior_c.hard_fitting(c, q_code_mu)
            # One-to-one cycle consistency
            cc_loss_c = self.prior_c.hard_fitting(c, c_gen)
            
        elif self.c_type == "discrete":
            # CE
            cc_loss_c = self.prior_c.cross_entropy_loss(c_logit if c_logit is not None else c, c_gen)
            NLL_loss_Q = self.prior_c.cross_entropy_loss(c_logit if c_logit is not None else c, q_code_mu)
           
           
            # Hard fitting # TODO 
            # if isinstance(self.prior_c, distributions.ContinuousCategorical):
            #     cc_loss_c = self.prior_c.hard_fitting(c_logit if c_logit is not None else c, c_gen)
            #     NLL_loss_Q = self.prior_c.hard_fitting(c_logit if c_logit is not None else c, q_code_mu)
            # else:
            #     raise NotImplementedError("")

        cc_loss_c *=0.05
        cc_loss = 0.15 * (cc_loss_z + cc_loss_c)
        return gen_loss + NLL_loss_Q + cc_loss, \
                [gen_loss, NLL_loss_Q, cc_loss_z, cc_loss_c]