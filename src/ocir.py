import torch.nn as nn
import torch.nn.functional as F
import torch 

from src.modules import (encoders, generators, discriminator_Q, distributions, flow_transforms) 


#TODO: gpu device, init, training, opt,  nf loss , testing the code, rul estimator and task, trajectory and task., visualization, metrics, etc. 
# CHeck nf-vae objective again.

class OCIR(nn.Module):
    def __init__(self, args, device): 
        super(OCIR, self).__init__()
        self.device = device
        self.c_type = args.c_type
        self.code_posterior_param = args.c_posterior_param
        # Prior Distributions p(z') and p(c)
        self.prior_z = distributions.DiagonalGaussian(args.dz, mean = 0, var = 1)
        self.prior_c = distributions.UniformDistribution() if args.c_type == "continuous" else distributions.DiscreteUniform(args.dc, onehot = True)
        
        # NF transform
        self.h = flow_transforms.LatentFlow(args, self.prior_z)
        # Encoders
        self.f_E = encoders.LatentEncoder(args, self.h) 
        self.f_C = encoders.CodeEncoder(args)
        
        # Generator and decoder
        self.G = generators.Decoder(args, self.h, self.prior_c)
        self.f_D = self.G
        
        # Discriminator and Q
        self.D = discriminator_Q.Discriminator(args)
        self.Q = discriminator_Q.CodePosterior(args)
    
    
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
    def L_R(self, x, tidx):
        mu, log_var = self.f_E(x, tidx)
        z, logdet, z0 = self.f_E.reparameterization_NF(mu, log_var)
            
        c, _ = self.f_C(x)
        
        
        x_rec = self.f_D(z,c)
        
        # Reconstruction & CC is implicitly made since G and f_D share the parameters
        recon = F.l1_loss(x_rec, x, reduction = 'mean') 

        # logq(z'|x), where z' = z0  # Check this objective again
        log_qz0 = self.prior_z.log_prob(z0, mu, log_var)
        log_qz = log_qz0 - logdet
        log_pz = self.prior_z.log_prob(z)
        # KL div & logdet
        kl_div = - (log_qz - log_pz).sum()
        
        return recon + kl_div
    
    def L_G_discriminator(self, x):
        sample_size = x.shape[0]
        x_gen, _ = self.G.generation(sample_size)
        fake = self.D(x_gen)
        real = self.D(x)
        
        real_loss = torch.mean((real -1)**2)
        fake_loss = torch.mean((fake)**2)
        return 0.5 * (real_loss + fake_loss)
    
    def L_G_generator(self, x):
        sample_size = x.shape[0]
        x_gen, set_letent_code = self.G.generation(sample_size)
        z, z0, c = set_letent_code
        
        gen = self.D(x_gen)
        
        q_code_mu, q_code_logvar = self.Q(x_gen)
        
        # CC
        mu, logvar = self.f_E(x_gen)
        c_gen, c_logvar = self.f_C(x_gen)
        
        # Generator loss
        gen_loss = 0.5 * torch.mean((gen - 1)**2)
        # MMI for Q, G
        NLL_loss_Q = self.prior_c.NLL_gau(c, q_code_mu, q_code_logvar, "sum")
        
        # CC
        cc_loss_z = self.prior_z.NLL(z0, mu, logvar, "mean")
        if self.c_type == "continuous":
            if self.code_posterior_param == "soft":
                cc_loss_c = self.prior_c.NLL_gau(c, c_gen, c_logvar, "sum")
            elif self.code_posterior_param == "hard":
                cc_loss_c = self.prior_c.hard_fitting(c, c_gen)
        elif self.c_type == "discrete":
            cc_loss_c = self.prior_c.cross_entropy_loss(c,c_gen)
        
        return gen_loss + NLL_loss_Q + (0.5 * (cc_loss_z + cc_loss_c))
    
    
    
    

    '''
        def forward(self, x):
        # Encode x into latent mean and variance
        h = self.encoder(x)
        mu, log_var = self.mu(h), self.log_var(h)

        # Reparameterization trick
        z0 = self.reparameterize(mu, log_var)
        log_qz0 = -0.5 * ((z0 - mu) ** 2 * torch.exp(-log_var) + log_var).sum(dim=-1)   --> self.gaussian.log_prob(z0, mu, log_var)

        # Apply normalizing flow transformations
        log_det_sum = 0
        z = z0
        for flow in self.flows:
            z, log_det = flow(z)
            log_det_sum += log_det

        # Compute KL term
        log_pz = -0.5 * (z ** 2).sum(dim=-1)  # Log prior p(z)   self.gaussian.log_prob(z)
        log_qz = log_qz0 - log_det_sum
        kl_div = (log_qz - log_pz).mean()

        # Decode
        x_recon = self.decoder(z)

        return x_recon, kl_div
        
        
        
        
        
        
        
        optimizer.zero_grad()

        x_recon, mu, log_var, log_det_jacobian = model(x)

        # 1. Reconstruction Loss
        reconstruction_loss = nn.functional.binary_cross_entropy(x_recon, x, reduction='sum') # or MSE, depending on data

        # 2. KL Divergence Loss
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        # 3. Jacobian Loss (added due to the NF)
        jacobian_loss = -torch.sum(log_det_jacobian) #negative because we want to maximize the log likelihood

        # Total Loss
        loss = reconstruction_loss + kl_loss + jacobian_loss

        loss.backward()
        optimizer.step()
    '''