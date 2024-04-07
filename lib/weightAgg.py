# 
# model parameter aggregation in FedReID
#

import copy
from threading import local
import torch
from lib.model import Generator
import numpy as np
import torch.nn as nn
from torch.autograd import Variable


def weights_aggregate(w_all,w_glob,idx_client):
    """
    w: client model parameters
    dp: differential privacy scale factor beta
    alpha_mu: update momentum in local weight aggregation alpha
    idx_client: selected local client index
    is_local: flag of local client
    return: aggregated model parameters

    shitong
    w[0]->global
    w[1]->local->expert
    """   
    for k in w_glob.keys():         
            # central server use the average of selected local clients for aggregation
            # low kl loss-> low model weight
        temp = torch.zeros_like(w_glob[k], dtype=torch.float32)
        for i in range(len(idx_client)):
            temp += w_all[idx_client[i]][k]
            # w_agg[k] = w_agg[k] + w_all[i][k] #* kl_weight[i]
        # privacy protection with differential privacy                                      
        temp = torch.div(temp, len(idx_client))
        w_glob[k].data.copy_(temp)   
        for i in range(len(idx_client)):
            w_all[idx_client[i]][k].data.copy_(temp)
            
        #shitong comment here
        # w_agg[k] = torch.div(w_agg[k], len(w_all)) + torch.mul(torch.randn(w_agg[k].shape), dp).type_as(w_agg[k])

    # for k in w_glob_G.keys():         
    #         # central server use the average of selected local clients for aggregation
    #         # low kl loss-> low model weight
    #     temp = torch.zeros_like(w_glob_G[k], dtype=torch.float32)
    #     for i in range(len(idx_client)):

    #         temp += w_all_G[idx_client[i]][k]
    #         # w_agg[k] = w_agg[k] + w_all[i][k] #* kl_weight[i]
    #     # privacy protection with differential privacy                                      
    #     temp = torch.div(temp, len(idx_client))
    #     w_glob_G[k].data.copy_(temp)   
    #     for i in range(len(idx_client)):
    #         w_all_G[idx_client[i]][k].data.copy_(temp)  
    
    return w_glob,w_all


def generator_distill(w_glob_G,w_all_G,opt):
    Generators = []
    for i in range(len(w_all_G)):
        generator =  Generator(opt).cuda()
        generator.load_state_dict(w_all_G[i])
        Generators.append(generator)
    server_G = Generator(opt).cuda()
    server_G.load_state_dict(w_glob_G)
    optimizer_G = torch.optim.Adam(server_G.parameters(), lr=0.0002, betas=(0.5, 0.999)) # default: lr = 0.0002
    criterion_mse = nn.MSELoss().cuda()
    epochs = 10  # TODO: can change later
    for epo in range(epochs):
        optimizer_G.zero_grad()
        # z = Variable(torch.cuda.FloatTensor(np.random.normal(0, 1, (now_batch_size, self.args.attr_num, self.args.latent_dim))))
        z = Variable(torch.cuda.FloatTensor(np.random.normal(0, 1, (opt.local_bs, opt.latent_dim))))

        gen_labels = Variable(torch.cuda.FloatTensor(np.random.uniform(size = (opt.local_bs,opt.attr_num)))) # generate -1,0 1 randomly 

        # gen_labels = label2one_hot(gen_labels,self.args.attr_opt).cuda()
        # gen_labels = label_gen() # TODO: add later, no need maybe

        gen_feat = server_G(z, gen_labels)
            # Loss measures generator's ability to fool the discriminator      
        # g_loss = self.criterion_mse(validity, valid)
 
        # local_feats = np.array([G(z,gen_labels).detach.cpu() for G in Generators])
        local_feats = torch.zeros((len(w_all_G),opt.local_bs,2048))
        for idx,G in enumerate(Generators):
            local_feats[idx] = G(z,gen_labels)
        local_feats = torch.mean(local_feats,dim=0).cuda()      
        g_mse_loss = criterion_mse(local_feats.detach(),gen_feat)
        # print('g_loss and CE_G_loss:',g_loss,CE_G_loss)
        g_mse_loss.backward()
        optimizer_G.step()
    return server_G.state_dict()

def weights_aggregate_server_MSE(w_all,w_glob,w_all_G,w_glob_G, opt,idx_client):
    
    for k in w_glob.keys():         
            # central server use the average of selected local clients for aggregation
            # low kl loss-> low model weight
        temp = torch.zeros_like(w_glob[k], dtype=torch.float32)
        for i in range(len(idx_client)):
            temp += w_all[idx_client[i]][k]
            # w_agg[k] = w_agg[k] + w_all[i][k] #* kl_weight[i]
        # privacy protection with differential privacy                                      
        temp = torch.div(temp, len(idx_client))
        w_glob[k].data.copy_(temp)   
        for i in range(len(idx_client)):
            w_all[idx_client[i]][k].data.copy_(temp)
        #shitong comment here
        # w_agg[k] = torch.div(w_agg[k], len(w_all)) + torch.mul(torch.randn(w_agg[k].shape), dp).type_as(w_agg[k])

    for k in w_glob_G.keys():         
            # central server use the average of selected local clients for aggregation
            # low kl loss-> low model weight
        temp = torch.zeros_like(w_glob_G[k], dtype=torch.float32)
        for i in range(len(idx_client)):

            temp += w_all_G[idx_client[i]][k]
                              
        temp = torch.div(temp, len(idx_client))
        w_glob_G[k].data.copy_(temp)   

    w_glob_G = generator_distill(w_glob_G,w_all_G,opt)
    for k in w_glob_G.keys(): 
        for i in range(len(idx_client)):
            w_all_G[idx_client[i]][k].data.copy_(w_glob_G[k])  
    
    return w_glob,w_all,w_all_G

