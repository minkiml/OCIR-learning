import torch.nn as nn
import torch.nn.functional as F
import torch 
import numpy as np
import src.modules as md # (encoders, generators, discriminator_Q, distributions, flow_transforms) 
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
class OCIR_deep(nn.Module):
    def __init__(self, 
                dx:int = 14, dz:int = 10, dc:int = 6, window:int = 25, d_model:int = 128, 
                num_heads:int = 4, z_projection:str = "aggregation", D_projection:str = "aggregation", 
                time_emb:bool = True, c_type:str = "discrete", c_posterior_param:str = "soft", encoder_E:str = "transformer",
                device = "cpu"): 
        super(OCIR_deep, self).__init__()
        self.device = device
        self.c_type = c_type
        self.z_projection = z_projection
        self.time_emb = time_emb
        self.code_posterior_param = c_posterior_param
        self.c2_projection = "spc"
        
        dc2 = 2
        # Prior Distributions p(z') and p(c)
        self.prior_z = md.DiagonalGaussian(dz, mean = 0, var = 1, device=device)
        # TODO there is distributional mismatch if use gasNLL or implement gaussian one for c 
        self.prior_c = md.UniformDistribution(device=device) if c_type == "continuous" \
           else md.DiscreteUniform(dc, onehot = True, device=device) 
                # else distributions.ContinuousCategorical(dc, 
                #                                         gumbel_temperature= 0.4,
                #                                         decay_rate= 0.95,
                #                                         dist="uniform",
                #                                         device=device)
                # else distributions.DiscreteUniform(dc, onehot = True, device=device) 

        # NF transform
        self.h = md.LatentFlow(dz, self.prior_z)
        
        # shared early layers betweew f_E and f_C
        self.shared_encoder_layers = transformers.SharedEncoder(dx=dx, dz=dz, window=window, d_model=d_model, 
                                          num_heads=num_heads,z_projection=z_projection, time_emb=time_emb, c2_projection=self.c2_projection)
        # Encoders
        self.f_E = md.LatentEncoder(dx=dx, dz=dz, window=window, d_model=d_model, 
                                          num_heads=num_heads, z_projection=z_projection, 
                                          time_emb=time_emb, encoder_E=encoder_E, p_h=self.h, shared_EC= True if self.shared_encoder_layers is not None else False) 
        
        self.f_C = md.CodeEncoder(dx=dx, dc=dc, d_model=d_model, 
                                        c_type=c_type, c_posterior_param=c_posterior_param, shared_EC= True if self.shared_encoder_layers is not None else False)
        
        self.prior_c2 = md.DiscreteUniform(dc2, onehot = True, device=device) 
           
        self.f_C2 = md.CodeEncoder_seq(dx=dx, dc=dc2, d_model=d_model, 
                                        c_type="discrete", c_posterior_param=c_posterior_param, 
                                        shared_EC= True if self.shared_encoder_layers is not None else False,
                                        c2_projection = self.c2_projection, window = window,
                                        time_emb = time_emb)
        # Generator and decoder
        self.G = md.Decoder(dx=dx, dz=dz, dc=dc, window=window, d_model=d_model,
                                    num_heads=num_heads, p_h = self.h, p_c = self.prior_c, p_c2 = self.prior_c2, dc2 = dc2)
        self.f_D = md.Decoder(dx=dx, dz=dz, dc=dc, window=window, d_model=d_model,
                                    num_heads=num_heads, p_h = self.h, p_c = self.prior_c, p_c2 = self.prior_c2, dc2 = dc2) #self.G # shared
        
        # Discriminator and Q
        self.shared_net = transformers.shared_transformer(dx=dx, d_model=d_model, window=window, 
                                                         num_heads=num_heads, D_projection=D_projection)
        # self.shared_net = None
        self.D = md.Discriminator(dx=dx, window=window, d_model=d_model,
                                               num_heads=num_heads, D_projection=D_projection,
                                               shared_layer= self.shared_net)
        self.Q = md.CodePosterior(dx=dx, dc=dc, d_model=d_model, c_type=c_type, 
                                               c_posterior_param=c_posterior_param,
                                               shared_layer=self.shared_net)

        self.Q2 = md.CodePosterior_seq(dx=dx, dc=dc2, d_model=d_model, c_type="discrete", 
                                               c_posterior_param=c_posterior_param,
                                               shared_layer=self.shared_net,
                                                c2_projection = self.c2_projection, window = window)
    
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
            annealing = min(0.1, epoch / 15)
        else: annealing = 0.1
        # annealing = 1.0
        N, L, _ = x.shape
        
        # beta is a warm-up coefficient for preventing negative 
        # Encoding
        if self.shared_encoder_layers is not None:
            h = self.shared_encoder_layers(x, tidx)
            
            hc = h
            hc2 = h
            if (self.z_projection == "spc") or (self.z_projection == "seq"):
                hc = hc[:,1:,:]
                hc2 = hc2[:,1:,:]
            if self.time_emb:
                hc = hc[:,:-1,:]
            if self.c2_projection == "spc":
                hc = hc[:,:-1,:]
                h = h[:,:-1,:]
            assert hc.shape[1] == L
        else: 
            h = x
            hc = x
            hc2 = x
        mu, log_var, zin = self.f_E(h, tidx)
        # z, logdet, z0 = self.f_E.reparameterization_NF(mu, log_var)
        
        eps = self.prior_z.sample(mu.shape)
        stds = torch.exp(0.5 * log_var)
        z0 = eps * stds + mu
        z, logdet, z0 = self.f_E.p_h(z0 = z0)
        
        c, _ = self.f_C(hc)
        c2 = self.f_C2(hc2)
        # Decoding
        x_rec = self.f_D(z,c, zin = zin, c2 = c2)
        x_rec_G = self.G(z.detach(),c.detach(), zin = zin, c2 = c2.detach())
        # Reconstruction & CC is implicitly made since G and f_D share the parameters
        recon = F.mse_loss(x_rec, x, reduction = 'sum') / (N)
        recon_G = F.mse_loss(x_rec_G, x, reduction = 'sum') / (N)
        # recon_G = None
        '''
        From normalizing flow, h , 
        log q(zk|x) = log q(z0|x) - sum_{k=1}^K log |det(df_k/dz_{k-1})|   , where zk  is z in our paper and z0 is z'
        
        KL(q(zk | x), p(z)) = E_{q(zk|x)}[log q(zk|x) - log p(z)]
                            = E_{q(z0|x)}[log q(z0|x) - sum_{k=1}^K log |det(df_k/dz_{k-1})| - log p(zk)],  with respect to q(z0|x)
                            -> -KL = E_{q(z0|x)}[- log q(z0|x) + sum_{k=1}^K log |det(df_k/dz_{k-1})| + log p(zk)]  --> we want to minimize it in full ELBO form  
        '''
        # # # logq(z0|x) 
        # log_qz0 = self.prior_z.log_prob(z0, mu, log_var)
        # # logq(zk|x) 
        # log_qz = log_qz0 - logdet
        
        # # logq(zK)
        # log_pz = self.prior_z.log_prob(z)
        # # KL div & logdet
        # kl_div =  log_qz - log_pz # .mean()
        # kl_div =  (log_qz-  log_pz).mean() # .mean()
        
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        log_qz0 = self.unit_MVG_Guassian_log_prob(eps)
        log_qz0 -= torch.sum(0.5 * log_var, dim=1)
        log_qz0 -= logdet
        log_p_z = self.unit_MVG_Guassian_log_prob(z)
        kl_div = torch.mean(log_qz0) - torch.mean(log_p_z)
        
        
        # print("logdet: ", logdet)
        # print(log_qz.mean())
        # print("logpz", log_pz.mean())
        # kl_div = torch.clamp(kl_div, min=0)
        
        # print("recon", recon)
        # print("kl_div", kl_div)
        
        # x 0.5 as the reconstruction is done through G (shared net) as well 
        return recon + (annealing*kl_div), [recon, annealing*kl_div, recon_G ]
    
    def L_G_discriminator(self, x):
        sample_size = x.shape[0]
        x_gen, _,_ = self.G.generation(sample_size)
        fake = self.D(x_gen)
        real = self.D(x)
        # print("D real:          ", real)
        # print("D fake:          ", fake)
        if real.dim() == 2:
            real_loss = torch.mean((real -1)**2)
            fake_loss = torch.mean((fake)**2) 
        elif real.dim() == 3:
            real_loss = torch.sum((real -1)**2, dim = 1)
            fake_loss = torch.sum((fake)**2, dim = 1) 
        
        # print("real_loss", real_loss)
        # print("fake_loss", fake_loss)

        return 0.5 * torch.mean(real_loss + fake_loss), [torch.mean(real_loss), torch.mean(fake_loss)] 
    
    def L_G_generator(self, x):
        sample_size, L, _ = x.shape
        x_gen, set_latent_samples, log_det = self.G.generation(sample_size) # TODO need to add reverse KL div
        z, z0, c, c_logit, c2, c2_logit = set_latent_samples #TODO yield logit samples when gumbel is used
        
        gen = self.D(x_gen)
        # print("gen:          ", gen)
        q_code_mu, q_code_logvar = self.Q(x_gen)
        q2_code_mu = self.Q2(x_gen)
        # CC
        if self.shared_encoder_layers is not None:
            h = self.shared_encoder_layers(x_gen)
            hc = h
            hc2 = h
            if (self.z_projection == "spc") or (self.z_projection == "seq"):
                hc = hc[:,1:,:]
                hc2 = hc2[:,1:,:]
            if self.time_emb:
                hc = hc[:,:-1,:]
            if self.c2_projection == "spc":
                hc = hc[:,:-1,:]
                h = h[:,:-1,:]
            assert hc.shape[1] == L
        else: 
            h = x
            hc = x
            hc2 = x
        mu, logvar, _ = self.f_E(h)
        c_gen, c_logvar = self.f_C(hc)
        c2_gen = self.f_C2(hc2)

        # TODO
        # reverse_kl = - log_det.mean()
        
        # Generator loss
        if gen.dim() == 2:
            gen_loss = 0.5 * torch.mean((gen - 1)**2)    
        elif gen.dim() == 3:
            gen_loss = 0.5 * torch.mean(torch.sum((gen - 1)**2, dim = 1))    
        
        # print("x_gen", x_gen)
        # print("q_code_mu", q_code_mu)
        # print("q_code_logvar", q_code_logvar) 
        
        # CC and MMI for Q, G  
        # eps = self.prior_z.sample(mu.shape)
        # stds = torch.exp(0.5 * logvar)
        # z_pred = eps * stds + mu
        # cc_loss_z = self.prior_z.hard_fitting(z0, z_pred)
        
        cc_loss_z = self.prior_z.NLL(z0, mu, logvar, "mean") # soft fitting
        # cc_loss_z = torch.tensor(0.)#self.prior_z.hard_fitting(z0, mu) # One-to-one cycle consistency  # TODO consider reparameterizing mu and doing the cc loss
        
        if self.c_type == "continuous":
            if self.code_posterior_param == "soft":
                # NLL
                cc_loss_c = self.prior_c.NLL_gau(c, c_gen, c_logvar, "mean")
                NLL_loss_Q = self.prior_c.NLL_gau(c, q_code_mu, q_code_logvar, "mean")
                
                # cc_loss_c2 = self.prior_c2.NLL_gau(c2, c2_gen, c2_logvar, "mean")
                # NLL_loss_Q2 = self.prior_c2.NLL_gau(c2, q2_code_mu, q2_code_logvar, "mean")

            elif self.code_posterior_param == "hard":
                # MSE
                cc_loss_c = self.prior_c.hard_fitting(c, c_gen)
                NLL_loss_Q = self.prior_c.hard_fitting(c, q_code_mu)
        
            # One-to-one cycle consistency
            # cc_loss_c = self.prior_c.hard_fitting(c, c_gen)
            
        elif self.c_type == "discrete":
            # CE
            cc_loss_c = self.prior_c.cross_entropy_loss(c_logit if c_logit is not None else c, c_gen)
            NLL_loss_Q = self.prior_c.cross_entropy_loss(c_logit if c_logit is not None else c, q_code_mu)
            
            
        cc_loss_c2 = self.prior_c2.cross_entropy_loss(c2_logit if c2_logit is not None else c2, c2_gen)
        NLL_loss_Q2 = self.prior_c2.cross_entropy_loss(c2_logit if c2_logit is not None else c2, q2_code_mu)
        
           
            # Hard fitting # TODO 
            # if isinstance(self.prior_c, distributions.ContinuousCategorical):
            #     cc_loss_c = self.prior_c.hard_fitting(c_logit if c_logit is not None else c, c_gen)
            #     NLL_loss_Q = self.prior_c.hard_fitting(c_logit if c_logit is not None else c, q_code_mu)
            # else:
            #     raise NotImplementedError("")
            
        # print("gen_loss", gen_loss)
        # print("NLL_loss_Q", NLL_loss_Q)
        # print("cc_loss_z", cc_loss_z)
        # print("cc_loss_c", cc_loss_c)
        return gen_loss + (NLL_loss_Q) + ((cc_loss_z + cc_loss_c)), \
                [gen_loss, NLL_loss_Q , cc_loss_z,  cc_loss_c, NLL_loss_Q2, cc_loss_c2]
    
    def unit_MVG_Guassian_log_prob(self, sample):
        return -0.5*torch.sum((sample**2 + np.log(2*np.pi)), dim=1)