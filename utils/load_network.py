#
# Load model
#

import torch
import os

from torchvision import models
import torch.nn as nn
# from lib.model_Gavin import resnet50


def load_network(network, save_path, gpu_ids,name):
    # model_pth = '{}'.format(opt.model_name) # saved model
    # name = opt.name
    # dir_name = os.path.join(opt.logs_dir, name)
    # save_path = os.path.join(dir_name, model_pth)
    name_model_dict = {
        'dukemtmc-reid':'model_0',
        'market1501':'model_1',
        'msmt17':'model_2',
        'cuhk03-np':'model_3'
        }    
    if name in name_model_dict.keys():
        with open(save_path, 'rb') as f:
            model_dict = network.state_dict()
            pretrain_dict = torch.load(f, map_location='cuda:%s'%gpu_ids[0])[name_model_dict[name]]
            # comment the following line if don't want the check
            pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
            network.load_state_dict(pretrain_dict)
            # network.load_state_dict(torch.load(f, map_location='cuda:%s'%gpu_ids[0]))#[name_model_dict[name]]) # map the model to the current GPU
            # network.load_state_dict(torch.load(f, map_location='cuda:%s'%gpu_ids[0])['local_model'])#[name_model_dict[name]]) # map the model to the current GPU

        print('loading network weight from {}'.format(name_model_dict[name]))
        return network
    else:
        with open(save_path, 'rb') as f:
            # network.load_state_dict(torch.load(f, map_location='cuda:%s'%gpu_ids[0]))#['local_model'])#['server_model']) # map the model to the current GPU
            network.load_state_dict(torch.load(f, map_location='cuda:%s'%gpu_ids[0])['server_model']) # map the model to the current GPU

        # print('loading network weight from server model') # map the model to the curren    t GPU   

        return network
 
def load_network_mergeImageNet(network, save_path, gpu_ids,name):
    # model_pth = '{}'.format(opt.model_name) # saved model
    # name = opt.name
    # dir_name = os.path.join(opt.logs_dir, name)
    # save_path = os.path.join(dir_name, model_pth)
    model_backbone = resnet50(pretrained=True)
    model_backbone.avgpool = nn.AdaptiveAvgPool2d((1,1))   


    with open(save_path, 'rb') as f:
        # network.load_state_dict(torch.load(f, map_location='cuda:%s'%gpu_ids[0]))#['local_model'])#['server_model']) # map the model to the current GPU
        checkpoint = torch.load(f, map_location='cuda:%s'%gpu_ids[0])['server_model'] # map the model to the current GPU
    

    pretrain_dict={}
    # pretrain_dict = {k: v for k, v in model_backbone.state_dict().items() if k in checkpoint.keys() and checkpoint[k].size() == v.size()}
    # import pdb
    # pdb.set_trace()
    for k,v in checkpoint.items():
        flag = 'bn' in k or 'downsample.1' in k
        if k[6:] in model_backbone.state_dict().keys() and model_backbone.state_dict()[k[6:]].size() == v.size() and flag:
            checkpoint[k]=model_backbone.state_dict()[k[6:]]
            # print(k[6:])
    # print('loading network weight from server model') # map the model to the curren    t GPU   
    # checkpoint.update(pretrain_dict)
    network.load_state_dict(checkpoint)
    return network
