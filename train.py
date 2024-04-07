# 
# Decentralised Person Re-Identification
# Shitong 

from __future__ import print_function, division
import torch
import numpy as np
import time

from lib.model import attribute_align_net,Generator,attribute_align_net_test
from config import opt
from utils.get_dataset import get_dataset
from lib.fedreid_train import FedReID_train

# from cvpr18xian
import random

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

torch.manual_seed(42)

def main(opt):
    # Set GPU
    # print(opt)
    str_ids = opt.gpu_ids.split(',')
    gpu_ids = []
    for str_id in str_ids:
        gid = int(str_id)
        if gid >=0:
            gpu_ids.append(gid)

    if len(gpu_ids)>0:
        torch.cuda.set_device(gpu_ids[0])

    print('Using GPU {} for model training'.format(gpu_ids[0]))


    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    np.random.seed(0) 
    # Prepare local client datasets
    print('----------Load client datasets----------')
    local_loaders,global_loader = get_dataset(opt, is_training=True)

    print('Loading data finished ')

    # num_ids_client = [] # number of training ids in each local client
    # for i in range(opt.nusers):
    #     num_ids_client.append(len(local_datasets[i].classes))
        

    # Model initialisation
    models = []

    for ids in range(opt.nusers):        
        model_idx = attribute_align_net(opt,local_loaders[ids]).cuda()  #list length 4=> will build 4 model here=> actually one model with 4 fully connected layers
        models.append(model_idx)                                   # when forward: set an parameter to choose the fully connected layer

    glob_model = attribute_align_net_test(opt,global_loader).cuda()
    print('Done')

    # Model training
    print('----------Training----------')
    model = FedReID_train(models, glob_model, opt, local_loaders, global_loader) # Central model

if __name__ == '__main__':
    main(opt)

