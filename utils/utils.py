import numpy as np
import torch
import time
import pdb


def scoring_func(global_model, local_model, dataloader, client_idx, args):
    global_model.eval()
    local_model.eval()

    udis_list = torch.tensor([]).cuda()
    udata_list = torch.tensor([]).cuda()

    with torch.no_grad():
        for _, (_, data) in enumerate(dataloader):
            image = data['image'].cuda()

            # surrogate global model
            g_logit = global_model(image)[0]
            g_logit = torch.clamp_max(g_logit, 80)
            alpha = torch.exp(g_logit) + 1
            total_alpha = torch.sum(alpha, dim=1, keepdim=True) # batch_size, 1, patch_size, patch_size

            g_pred = alpha / total_alpha
            g_entropy = torch.sum(- g_pred * torch.log(g_pred), dim=1)     
            g_u_data = torch.sum((alpha / total_alpha) * (torch.digamma(total_alpha + 1) - torch.digamma(alpha + 1)), dim=1)
            
            g_u_dis = g_entropy - g_u_data
            udis_list = torch.cat((udis_list, g_u_dis.mean(dim=[1,2])))

            # local model
            l_logit = local_model(image)[0]
            l_logit = torch.clamp_max(l_logit, 80)
            alpha = torch.exp(l_logit) + 1
            total_alpha = torch.sum(alpha, dim=1, keepdim=True) # batch_size, 1, patch_size, patch_size
            l_u_data = torch.sum((alpha / total_alpha) * (torch.digamma(total_alpha + 1) - torch.digamma(alpha + 1)), dim=1)

            udata_list = torch.cat((udata_list, l_u_data.mean(dim=[1,2])))

    return udis_list.mean().cpu().numpy(), udata_list.mean().cpu().numpy()

