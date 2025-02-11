import torch
import torch.nn as nn
from src.modules.flow_transforms import LatentFlow
from src.modules import distributions
from src.ocir import OCIR
    
if __name__ == '__main__':
    
    
    
    from types import SimpleNamespace
    def model_size(m, model_name):
        print(f"Model: {model_name}")
        total_param = 0
        for name, param in m.named_parameters():
            num_params = param.numel()
            total_param += num_params
            print(f"{name}: {num_params} parameters")
        
        print(f"Total parameters in {model_name}: {total_param}")
        print("")
        return total_param
    
    args = SimpleNamespace(**{"dz": 16,
                              "dc": 6,
                              "dx": 3,
                              "c_type": "discrete",
                              "window": 25,
                              "d_model": 64,
                              "encoder_E": "transformer",
                              "z_projection": "aggregation",
                              "D_projection": "aggregation",
                              "c_posterior_param": "soft",
                              "num_heads": 4
                                        })
    
    m = OCIR(args, "cpu")
    device = torch.device(f'cuda:{2}' if torch.cuda.is_available() else 'cpu')
    print(f"GPU (device: {device}) used" if torch.cuda.is_available() else 'cpu used')
        
    # print(m)
    # model_size(m, "ocir")
    N = 3
    x = torch.randn((N,args.window, args.dx))
    tidx = torch.arange(N).reshape(N, 1)
    
    ''' fe '''
    # mu, log_var = m.f_E(x, tidx)
    # z, ldj, z0 = m.f_E.reparameterization_NF(mu, log_var)
    # print(mu)
    # print(mu.shape)
    # print("z", z)
    # print("z", z.shape)
    
    # print("ldj", ldj)
    # print("ldj", ldj.shape)
    
    # print("z0", z0)
    # print("z0", z0.shape)
    
    ''' fc '''
    # c, c_logvar  = m.f_C(x)
    # print(c)
    # print(c.shape)
    # print(c_logvar)
    # print(c_logvar.shape)    
    
    ''' fd & G'''
    z = m.f_E.encoding(x, tidx)
    c, _ = m.f_C(x)
    
    x_rec = m.f_D(z, c)
    # print(x_rec)
    print("f_D(z)", x_rec.shape)
    x_gen, set = m.f_D.generation(N)
    # print(x_gen)
    print("G(x|z,c)",x_gen.shape)
    z,z0, c = set
    # print(c)
    print("p(c)",c.shape)
    
    
    ''' discrimintor and Q'''
    
    score = m.D(x)
    c, _ = m.Q(x)
    # print(score)
    print("score", score.shape)
    # print(c)
    print("Q(c|x)", c.shape)
    
    # TODO: check gpu setup, rnn arg, test spc 

    # prior_z = distributions.DiagonalGaussian(args.dz, mean = 0, var = 1)
    # prior_c = distributions.UniformDistribution() 
    # prior_c_disc = distributions.DiscreteUniform(args.dc, onehot = True)
    
    # m = LatentFlow(args, prior_z)
    
    # N = 3
    
    # xx = torch.randn((N,3, args.dz))
    
    # # yy, ldj, z_0 = m(xx)
    
    # # zz, _ = m.inverse(yy)
    
    # # if torch.allclose(z_0, zz, rtol = 1e-4):
    # #     print("invertible")
        
    # xx = torch.randn((N,3, args.dc))
    # print(prior_c.sample(xx))
    # print(prior_c.sample(xx, target = 0.2).shape)
    
    # print(prior_c_disc.sample(xx))
    # print(prior_c_disc.sample(xx, target = 2))