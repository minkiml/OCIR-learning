import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import os
import time

from util_modules import utils
from datasets.cmapss import load_CMAPSS
from datasets.circles import load_circle

from src.ocir import OCIR 

class DotDict(dict):
    """
    a dictionary that supports dot notation 
    as well as dictionary access notation 
    usage: d = DotDict() or d = DotDict({'val1':'first'})
    set attributes: d.val2 = 'second' or d['val2'] = 'second'
    get attributes: d.val2 or d['val2']
    """
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, dct):
        for key, value in dct.items():
            if hasattr(value, 'keys'):
                value = DotDict(value)
            self[key] = value

class Solver(object):
    DEFAULTS = {}
    def __init__(self, config, logger):
        super(Solver, self).__init__()
        self.__dict__.update(Solver.DEFAULTS, **config)
        self.logger = logger
        
        seed = self.seed
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.device = torch.device(f'cuda:{self.gpu_dev}' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"GPU (device: {self.device}) used" if torch.cuda.is_available() else 'cpu used')
        
        
    def build_dataset(self):
        # Dataset instantiation
        if self.dataset == "circle": 
            raise NotImplementedError("")
            load_circle()
        else: 
            self.training_data, self.testing_data, self.val_data = load_CMAPSS(dataset = self.dataset,
                                                                    data_path = self.data_path,
                                                                    task = self.task,
                                                                    T = self.T,
                                                                    H = self.H,
                                                                    H_lookback = self.hyper_lookback,
                                                                    rectification = 125,
                                                                    batch_size = self.batch_size,
                                                                    normalize_rul = True,
                                                                    logger = self.logger)
        
        self.ipe = len(self.training_data)
        
    def build_model(self):
        
        ocir = OCIR(1, self.device) # TODO
        self.ocir = utils.load_model(ocir, self.model_save_path, "ocir")
        self.ocir.to(self.device)
        
    def get_optimizers(self):
        raise NotImplementedError("")
    
    def early_stop(self):
        raise NotImplementedError("")
    