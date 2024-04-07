# 
# local client datasets loading and partition
#

import torch
from torchvision import datasets, transforms
import os
from utils.sampling import partition
from utils.attribute_dataset import attribute_dataset

import torch

import cvpr18xian.util as util



# Loading data with torchvision and torch.utils.data packages
def get_dataset(opt, is_training=True):
    # load training data
    if is_training:
        # Local client training data
        local_loaders = []
        global_loader = util.DATA_LOADER(opt)
        #iid from one dataset
        # random_perm = torch.randperm(global_loader.ntrain)

        # for idx in range(opt.nusers):
        #     local_datasets.append(util.random_loader(opt,idx,random_perm))

        #non iid setting from one dataset
        random_class_perm=torch.randperm(global_loader.seenclasses.size(0))
        
        
        random_class = global_loader.seenclasses[random_class_perm]
        for idx in range(opt.nusers):
            local_loaders.append(util.class_split_loader(opt,idx,random_class))

        return local_loaders,global_loader
    # load testing data
    # else:
    #     data_transforms = transforms.Compose([
    #             # transforms.Resize((288,144), interpolation=3),
    #             transforms.Resize((256,128), interpolation=3),

    #             transforms.ToTensor(),
    #             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    #             ])

    #     image_datasets = {x: datasets.ImageFolder( os.path.join(opt.test_data_dir, x), data_transforms) for x in ['gallery','query']}
    #     dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
    #                                              shuffle=False, num_workers=16) for x in ['gallery','query']}

    #     return image_datasets, dataloaders

