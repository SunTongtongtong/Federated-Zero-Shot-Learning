# 
# Federated person re-identification for decentralised model learning
# Guile Wu and Shaogang Gong
# 2020
#

from __future__ import print_function, division
import torch
import numpy as np
import time
import copy
import sys
import os

from utils.logging import Logger
from lib.weightAgg import *
from lib.localUpdate import LocalUpdateLM
from torch.utils.tensorboard import SummaryWriter
from utils.meters import AverageMeter
from lib.localUpdate import generate_syn_feature
import cvpr18xian.classifier2 as classifier2
import cvpr18xian.util as util
from tSNE import visualize_tSNE, visualize_tSNE_2label
from confusion_matrix import viusal_confusion_matrix



def FedReID_train(models, glob_model, opt, local_dataloaders,global_loader):
    # Model save directory
    writer = SummaryWriter('runs/{}'.format(opt.name) + time.strftime(".%m_%d_%H:%M:%S"))
    w_glob = glob_model.state_dict() # weights of neurons

    name = opt.name
    dir_name = os.path.join(opt.logs_dir, name)
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)
    print('===> Datasets:',opt.dataset)

    model_pth = 'model_{}'.format(opt.name)+ time.strftime(".%m_%d_%H:%M:%S") + '.pth' # saved model
    model_saved = os.path.join(dir_name, model_pth) # model directory
    sys.stdout = Logger(os.path.join(dir_name, 'log' + time.strftime(".%m_%d_%H:%M:%S") + '.txt')) # training log

    since = time.time() # training start time
    num_epochs = opt.global_ep # global communication epochs
    m = max(int(opt.frac * opt.nusers), 1) # selected client number in each global communication epoch (1 =< C*N <= N)

    w_all = []#, w_tmp = [], [] # all local model parameters and local temporal model parameters
    # w_all_G = []
    for i in range(opt.nusers):
        w_all.append(models[i].state_dict()) # initial
        # w_all_G.append(models[i].netG.state_dict())
    loss_meter = AverageMeter('Total loss for selected clients', ':6.3f')


    best_H=0
    best_acc=0

    # Training
    for epoch in range(num_epochs):
        print('Global Training Epoch {}/{}'.format(epoch+1, num_epochs))
        loss_meter.reset() 
        idxs_users_selected = np.random.choice(range(opt.nusers), m, replace=False) # randomly selected clients   # TODO: change later 
        # idxs_users_selected = np.array([2])
        # local client model updating
        for idx in idxs_users_selected:

            print('=====training global round {} for client {} =============='.format(epoch,idx))
            # local client model initialisation and local dataset partition
            # idxs: each local client only contains one user here
            local = LocalUpdateLM(args=opt, data_loader=local_dataloaders[idx], nround=epoch)

            # local client weight update
            models[idx].load_state_dict(w_all[idx])
            # models[idx].netG.load_state_dict(w_all_G[idx])
            # local client model training, return model parameters and training loss
            out_dict = local.update_weights(model=copy.deepcopy(models[idx]), cur_epoch=epoch,idx_client=idx)

            # store updated local client parameters
            # loss_meter.update(out_dict['loss_meter'])
            #shitong

            # store all local client parameters (some clients are not updated in the randomly selection)
            w_all[idx] = copy.deepcopy(out_dict['model_params'])
            # w_all_G[idx] = copy.deepcopy(out_dict['G_params'])

            if opt.gzsl:         
                writer.add_scalar('client {} seen acc'.format(idx),
                                out_dict['acc_seen'],
                                epoch)
                writer.add_scalar('client {} H'.format(idx),
                        out_dict['H'],
                        epoch)

            writer.add_scalar('client {} unseen acc'.format(idx),
                out_dict['acc_unseen'],
                epoch)  

        # central server model updating 
        if opt.agg == 'avg': # current version  only supports modified federated average strategy
            w_glob,w_all= weights_aggregate(w_all, w_glob, idx_client=idxs_users_selected)#  central model parameter update
            # w_glob,w_all,w_all_G = weights_aggregate_server_MSE(w_all, w_glob,w_all_G,w_glob_G, opt,idx_client=idxs_users_selected)#  central model parameter update
        
        # For training the global model
        glob_model.load_state_dict(w_glob)

        print('======> GLOBAL server performance===>')

        syn_feature, syn_label = generate_syn_feature(glob_model,global_loader.unseenclasses, global_loader.attribute, opt.syn_num,opt,GZSL=False)

        # Zero-shot learning
        # syn_feature, syn_label = generate_syn_feature(glob_model, global_loader.unseenclasses, global_loader.attribute, opt.syn_num,opt) 
        cls = classifier2.CLASSIFIER(syn_feature, util.map_label(syn_label, global_loader.unseenclasses), global_loader, global_loader.unseenclasses.size(0), opt.cuda, opt.classifier_lr, 0.5, 25, opt.syn_num, False,TF_label_flag=True)
        acc = cls.acc
        if cls.acc>=best_acc:
            best_acc = cls.acc
            best_y_pred, best_y_real = cls.best_y_pred, cls.best_y_real

            best_syn_feature = syn_feature 
            best_epoch = epoch  
            best_syn_label = syn_label  
            print('ZSL BEST:===> unseen class accuracy= ', best_acc)
        print('ZSL unseen class accuracy= ', acc)

     
        gzsl_syn_feature, gzsl_syn_label = generate_syn_feature(glob_model,torch.cat((global_loader.seenclasses, global_loader.unseenclasses), dim=0), global_loader.attribute, opt.syn_num,opt,GZSL=True)
        # FL cannot use local client seen features, generate all seen features
        # train_X = torch.cat((global_loader.train_feature, syn_feature), 0)
        # train_Y = torch.cat((global_loader.train_label, syn_label), 0)
        nclass = opt.nclass_all
        cls = classifier2.CLASSIFIER(gzsl_syn_feature, gzsl_syn_label, global_loader, nclass, opt.cuda, opt.classifier_lr, 0.5, 25, opt.syn_num, True)
        if cls.H>=best_H:
            best_H=cls.H
            best_seen=cls.acc_seen
            best_unseen = cls.acc_unseen
            # best_TF = cls.TF_label
            print('GZSL BEST:===>  unseen=%.4f, seen=%.4f, h=%.4f' % (best_unseen, best_seen, best_H))
        
        print('GZSL unseen=%.4f, seen=%.4f, h=%.4f' % (cls.acc_unseen, cls.acc_seen, cls.H))


    category = [name[0][0] for name in global_loader.unseenclassnames ]
    
    # with open(opt.dataset+'_flclswgan_SMA_SKA_cfs.npy','wb') as f:
    #     np.save(f, best_y_pred.cpu().detach().numpy())
    #     np.save(f,best_y_real.cpu().detach().numpy())
    # if opt.dataset=='SUN' or opt.dataset=='CUB':
    #     viusal_confusion_matrix(best_y_pred,best_y_real,category,opt.dataset+'confusion',showvalue=False)
    # else:
    #     viusal_confusion_matrix(best_y_pred,best_y_real,category,opt.dataset+'confusion')


    print('GZSL BEST:===>  unseen=%.4f, seen=%.4f, h=%.4f' % (best_unseen, best_seen, best_H))
    print('ZSL BEST:===> unseen class accuracy= ', best_acc)
    return models
