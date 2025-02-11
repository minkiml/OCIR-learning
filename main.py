'''Base main script for easy working'''

import os
import argparse
import logging
import numpy as np
import torch

from torch.backends import cudnn
from util_modules import utils

def main():
    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_path", type=str, default='./Classification/Logs_/logs_', help="path to save all the products from each trainging")
    parser.add_argument("--id_", type=int, default=0, help="Run id")
    parser.add_argument("--data_path", type=str, default='./data', help="path to grab data")
    parser.add_argument("--description", type=str, default='', help="optional")
    
    # Task & data args
    parser.add_argument("--task", type=str, default="RL", choices = ["RL", "rul_estimation", 
                                                                    "rep_level_trj", "data_level_trj"])
    parser.add_argument("--dataset", type=str, default="FD001", choices = {"FD001", "FD002", "FD003", "FD004", "circle"})
    parser.add_argument("--c_type", type=str, default="discrete", help = " This argument is for circle data")
    parser.add_argument("--n_epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--patience", type=int, default=10)
    
    # Save path
    parser.add_argument('--model_save_path', type=str, default='checkpoints')
    parser.add_argument('--plots_save_path', type=str, default='plots')
    parser.add_argument('--his_save_path', type=str, default='hist')
    
    # Training args
    parser.add_argument("--seed", type=int, default=55)
    parser.add_argument("--gpu_dev", type=str, default="6")
    
    
    # Model args
    parser.add_argument("--window", type=int, default=25)
    parser.add_argument("--H", type=int, default=2)
    parser.add_argument("--hyper_lookback", type=int, default=2)
    # parser.add_argument("--ocs_dim", type=int, default=1) num_heads
    
    parser.add_argument("--dx", type=int, default=14)
    parser.add_argument("--dz", type=int, default=10)
    parser.add_argument("--dc", type=int, default=1, help = "if discrete, it is the expected number of ocs")
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=4)
      
    # Latent encoder
    parser.add_argument("--encoder_E", type=str, default="transformer", choices= ["transformer", "TCN"])
    parser.add_argument("--z_projection", type=str, default="aggregation", 
                        choices = ["aggregation", "spc", "rnn"])
    # Discriminator
    parser.add_argument("--D_projection", type=str, default="aggregation", 
                    choices = ["aggregation", "spc",  "rnn"])
    # Q and coder encoder
    parser.add_argument("--c_posterior_param", type=str, default="soft", choices = ["soft", "hard"])
    ...
    
    parser.add_argument("--trajectory_net", type=str, default="")
    ...
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
    
    # Check args


    args = vars(config)
    print('------------ Options -------------')
    for k, v in sorted(args.items()):
        logger.info('%s: %s' % (str(k), str(v)))
    print('-------------- End ----------------')
    
    main(config)
    
    
    # Training config
    
    # opt config
    
    # Data config
    
    # Model config
    
    #  

    # if config.task == "training":
    #     logger.info(f"********************* Training Pipeline *********************")
    #     training_pipeline()
        
    # elif config.task == "rul_estimation":
    #     logger.info(f"********************* RUL estimation Pipeline *********************")
    #     rul_estimation_pipeline()
    
    # elif config.task == "rep_level_trj":
    #     logger.info(f"********************* Representation level trajectory construction Pipeline *********************")
    #     rep_level_trajectory_pipeline()
    
    # elif config.task == "data_level_trj":
    #     logger.info(f"********************* Data level trajectory construction Pipeline  *********************")
    #     data_level_trajectory_pipeline()