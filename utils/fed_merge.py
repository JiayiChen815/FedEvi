import pdb
import numpy as np
import copy
import torch

def dict_weight(dict1, weight):
    for k, v in dict1.items():
        dict1[k] = weight * v
    return dict1
    
def dict_add(dict1, dict2):
    for k, v in dict1.items():
        dict1[k] = v + dict2[k]
    return dict1


def FedAvg(global_model, local_models, client_weight, bn=True):
    new_model_dict = None

    if bn:
        for client_idx in range(len(local_models)):
            local_dict = local_models[client_idx].state_dict()

            if new_model_dict is None:  # init
                new_model_dict = dict_weight(local_dict, client_weight[client_idx])
            else:
                new_model_dict = dict_add(new_model_dict, dict_weight(local_dict, client_weight[client_idx]))
        global_model.load_state_dict(new_model_dict)

    else:
        for key in global_model.state_dict().keys():
            if 'bn' not in key:
                temp = torch.zeros_like(global_model.state_dict()[key], dtype=torch.float32)
                for client_idx in range(len(client_weight)):
                    temp += client_weight[client_idx] * local_models[client_idx].state_dict()[key]
                global_model.state_dict()[key].data.copy_(temp)
            
    return global_model


def FedUpdate(global_model, local_models, bn=True):
    if bn:
        global_dict = global_model.state_dict()
        for client_idx in range(len(local_models)):
            local_models[client_idx].load_state_dict(global_dict)
    else:
        # pdb.set_trace()
        for key in global_model.state_dict().keys():
            if 'bn' not in key:
                for client_idx in range(len(local_models)):
                    local_models[client_idx].state_dict()[key].data.copy_(global_model.state_dict()[key])

    return local_models


