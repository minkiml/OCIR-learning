'''Base main script for easy working'''

import os
import argparse
import logging
from copy import deepcopy 
from torch.backends import cudnn
from util_modules import utils
import pipelines
import time
def Learning_RL(args, logger):
    logger.info(f"********************* Representation learning *********************")
    if config.net == "ocir":
        learning = pipelines.RlPipeline(args, logger)
    elif config.net == "infogan": 
        learning = pipelines.InfoGANPipeline(args, logger)
    elif config.net == "vae": 
        learning = pipelines.VAEPipeline(args, logger)
    elif config.net == "ae": 
        learning = pipelines.AEPipeline(args, logger)
    elif config.net == "ocir_deep":
        learning = pipelines.RlPipeline_deep(args, logger)
    model = learning()
    # pipelines.Stationarization(args, model, logger) # TODO ocir only right now 
    return model

def Learning_RUL(args, logger, encoder = None, shared_layer = None):
    logger.info(f"********************* RUL estimation Pipeline *********************")
    learning = pipelines.RulPipeline(args, logger, encoder, shared_layer)
    model = learning()
    return model

def Learning_TRJ(args, logger, ocir):
    if config.task == "data_trj":
        logger.info(f"********************* Data level trajectory construction Pipeline  *********************")
        learning = pipelines.DataTrjPipeline(args, logger, ocir)
        model = learning()
        return model
    else:
        raise NotImplementedError("")
    
def main(args, logger):
    if config.task == "RL":
        rl_model = Learning_RL(args, logger)
        
    elif config.task == "rul":
        # encoder = None
        # shared_layer = None
        # if os.path.exists(os.path.join(config.model_save_path, 'rul_cp.pth')):
        #     pass
        # else:
        #     # Load a traied encoder if available, otherwise will train one first 
        #     rl_model = Learning_RL(args, logger)
        #     encoder = deepcopy(rl_model.f_E)
        #     if rl_model.shared_encoder_layers:
        #         shared_layer = deepcopy(rl_model.shared_encoder_layers)
        rl_model = Learning_RL(args, logger)
        rul_model = Learning_RUL(args, logger, rl_model, None)
        
    elif (config.task == "data_trj"):
        start = time.time()
        rl_model = Learning_RL(args, logger)
        print("time taken: ",time.time() - start)
        # rul_model = Learning_RUL(args, logger, rl_model, None)
        trj_model = Learning_TRJ(args, logger, rl_model)
        
    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_path", type=str, default='./Logss/logs_', help="path to save all the products from each trainging")
    parser.add_argument("--id_", type=int, default=0, help="Run id")
    parser.add_argument("--data_path", type=str, default='./datasets/cmapss_dataset', help="path to grab data")
    parser.add_argument("--description", type=str, default='test_run', help="optional")
    
    # Task & data args
    parser.add_argument("--task", type=str, default="RL", choices = ["RL", "rul", 
                                                                    "stationarization", "data_trj", "all"])
    parser.add_argument("--dataset", type=str, default="FD001", choices = {"FD001", "FD002", "FD003", "FD004", "circle"})
    parser.add_argument("--c_type", type=str, default="discrete", help = " This argument is for circle data")
    parser.add_argument("--net", type=str, default="ocir", help = ["ocir", "infogan", "vae", "ae", "ocir_deep"])
    parser.add_argument("--valid_split", type=float, default=0., help= "learning rate")
    
    # Training args
    parser.add_argument("--seed", type=int, default=55)
    parser.add_argument("--gpu_dev", type=str, default="6")
    parser.add_argument("--n_epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--patience", type=int, default=10)
    
    parser.add_argument("--rul_epochs", type=int, default=50)
    parser.add_argument("--fore_epochs", type=int, default=50)
    parser.add_argument("--fore_batch", type=int, default=128)
    # optimizer
    parser.add_argument("--lr_", type=float, default=5e-4, help= "learning rate")
    parser.add_argument("--rul_lr", type=float, default=5e-4, help= "learning rate for rul")
    parser.add_argument("--trj_lr", type=float, default=5e-4, help= "learning rate for trj")
    parser.add_argument("--scheduler", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--kl_annealing", type=float, default=0.15)

    ## Scheduler params
    parser.add_argument("--warm_up", type=float, default=0.2, help="portion of warm up given number of epoches, e.g., 20 percent by defualt")
    parser.add_argument("--start_lr", type=float, default=1e-5, help="starting learning rate")
    parser.add_argument("--ref_lr", type=float, default=1.5e-4, help= "peak learning rate")
    parser.add_argument("--final_lr", type=float, default=1e-5, help = "final learning rate")
    parser.add_argument("--start_wd", type=float, default=0.01, help = "starting weight decay, setting it to 0. means no decay and decay scheduler")
    parser.add_argument("--final_wd", type=float, default=0.0001, help = "fianl weight decay")
    # TODO consider specifying separate lr 
    # Save path
    parser.add_argument('--model_save_path', type=str, default='checkpoints')
    parser.add_argument('--plots_save_path', type=str, default='plots')
    parser.add_argument('--his_save_path', type=str, default='hist')
    
    # Model args
    parser.add_argument("--window", type=int, default=25)
    parser.add_argument("--H", type=int, default=2)
    parser.add_argument("--hyper_lookback", type=int, default=2)
    parser.add_argument("--time_embedding", type=bool, default=False)
    parser.add_argument("--c_kl", type=int, default=0, help = "kl div for f_C")
    
    parser.add_argument("--dx", type=int, default=14)
    parser.add_argument("--dz", type=int, default=20)
    parser.add_argument("--dc", type=int, default=6, help = "if discrete, it is the expected number of ocs")
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=2)
    
    ## Latent encoder
    parser.add_argument("--encoder_E", type=str, default="transformer", choices= ["transformer", "TCN"])
    parser.add_argument("--z_projection", type=str, default="spc", 
                        choices = ["aggregation", "spc", "rnn", "seq", "aggregation_all"])
    ## Discriminator
    parser.add_argument("--D_projection", type=str, default="spc", 
                    choices = ["aggregation", "spc",  "rnn", "None"])
    ## Q and coder encoder
    parser.add_argument("--c_posterior_param", type=str, default="soft", choices = ["soft", "hard"])
    
    # Downstream task hyper params
    parser.add_argument("--extractor", type=str, default="ocir", choices= ["ocir", "vae"])
    parser.add_argument("--trajectory_net", type=str, default="")
    ...
    
    # Baseline hyperparams
    parser.add_argument("--conditional", type=int, default=1)
    
    
    config = parser.parse_args()
    
    log_path = config.log_path + config.dataset + "_" + f"{config.id_}" + "_" + f"{config.description}" 
    
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    os.environ["log_loc"] = f"{log_path}"
    root_dir = os.getcwd() 
    logging.basicConfig(filename=os.path.join(root_dir, f'{log_path}/log_all.txt'), level=logging.INFO,
                        format = '%(asctime)s - %(name)s - %(message)s')
    logger = logging.getLogger('In main')
    
    config.model_save_path = os.path.join(log_path,"checkpoints") 
    utils.mkdir(config.model_save_path)
    config.plots_save_path = os.path.join(log_path,"plots") 
    utils.mkdir(config.plots_save_path)
    config.his_save_path = os.path.join(log_path,"hist") 
    utils.mkdir(config.his_save_path)
    
    config.time_embedding = False
    # Check args
    if (config.dataset == "FD001") or (config.dataset == "FD003"):
        config.c_type = "continuous"
    else:
        config.c_type = "discrete"
        # config.dc = 6
    
    args = vars(config)
    print('------------ Options -------------')
    for k, v in sorted(args.items()):
        logger.info('%s: %s' % (str(k), str(v)))
    print('-------------- End ----------------')
    
    main(args, logger)