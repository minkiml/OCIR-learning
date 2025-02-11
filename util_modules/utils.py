import json
import os
import torch
import numpy as np

def mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def save_args(params, filename, path):
    print("saved the new arguments")
    path_ = os.path.join(path, filename)
    with open(path_+".json", 'w') as f:
        json.dump(vars(params), f, indent=4)
        
def save_model(model_, path_, name):
    model_.train(True)
    torch.save(model_.state_dict(), os.path.join(path_, f"{name}_cp.pth"))

def load_model(model, path_, name):
    try:
        if os.path.exists(os.path.join(path_, f'{name}_cp.pth')):
            print("Pre-trained model is loaded")
            return model.load_state_dict(torch.load(
                    os.path.join(path_, f'{name}_cp.pth')))
        else:
            print("No pre-trained model exists")
            return model
    except: 
        return None



def zeroout_gradient(layers_):
    # Zero out the gradients in layer b computed with respect to out_B 
    for param in layers_.parameters():
        if param.grad == None:
            # To prevent having None value in grad in the beginning of training
            pass
        else:
            param.grad = torch.zeros_like(param.grad)
