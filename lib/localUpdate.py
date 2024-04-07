# 
# Local client training in FedReID for decentralised person re-identification
#

import torch
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import lr_scheduler
from lib.kl_loss import KLLoss

import time
from tqdm import tqdm

from utils.meters import AverageMeter
import numpy as np
from utils.utils import label2one_hot

import cvpr18xian.classifier as classifier
import cvpr18xian.classifier2 as classifier2

import cvpr18xian.util as util
import torch.autograd as autograd


# local client dataset partition


def calc_gradient_penalty(opt,netD, real_data, fake_data, input_att):
    #print real_data.size()
    alpha = torch.rand(opt.batch_size, 1)
    alpha = alpha.expand(real_data.size())
    if opt.cuda:
        alpha = alpha.cuda()

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    if opt.cuda:
        interpolates = interpolates.cuda()

    interpolates = Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates, Variable(input_att))

    ones = torch.ones(disc_interpolates.size())
    if opt.cuda:
        ones = ones.cuda()

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=ones,
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * opt.lambda1
    return gradient_penalty


class DatasetSplitLM(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        return self.dataset[self.idxs[item]]


def generate_syn_feature(glob_model, classes, attribute, num,opt,test=True,GZSL=False):
    netG = glob_model.netG
    nclass = classes.size(0)
    syn_feature = torch.FloatTensor(nclass*num, opt.resSize)
    syn_label = torch.LongTensor(nclass*num) 
    syn_att = torch.FloatTensor(num, opt.attSize+512)
    syn_noise = torch.FloatTensor(num, opt.nz)
    if opt.cuda:
        syn_att = syn_att.cuda()
        syn_noise = syn_noise.cuda()
        
    for i in range(nclass):
        # import pdb
        # pdb.set_trace()
        iclass = classes[i]          

        if opt.mergeMapping:
            iclass_att = attribute[iclass].cuda()
            clip_encoder = glob_model.clip_encoder
            # clip_att = clip_encoder([torch.tensor(i)],torch.unsqueeze(iclass_att,0),test=True)
            # # clip_att = clip_encoder([iclass],torch.unsqueeze(iclass_att,0))
            # clip_att = clip_att.repeat(num, 1)

            clip_att = clip_encoder(torch.tensor(i).repeat(num),torch.unsqueeze(iclass_att,0).repeat(num,1),test=True,GZSL=GZSL) 
            # clip_att = clip_encoder(torch.tensor(iclass).repeat(num),torch.unsqueeze(iclass_att,0).repeat(num,1),test=True,GZSL=GZSL) 

            syn_att.copy_(clip_att)
        else:
            iclass_att = attribute[iclass]
            syn_att.copy_(iclass_att.repeat(num, 1))
        syn_noise.normal_(0, 1)
        output = netG(Variable(syn_noise, volatile=True), Variable(syn_att, volatile=True))
        syn_feature.narrow(0, i*num, num).copy_(output.data.cpu())
        syn_label.narrow(0, i*num, num).fill_(iclass)
    return syn_feature, syn_label


def generate_seen_syn_feature(glob_model, classes, attribute, num,opt,test=True):
            netG = glob_model.netG
            nclass = classes.size(0)
            syn_feature = torch.FloatTensor(nclass*num, opt.resSize)
            syn_label = torch.LongTensor(nclass*num) 
            syn_att = torch.FloatTensor(num, opt.attSize)
            syn_noise = torch.FloatTensor(num, opt.nz)
            if opt.cuda:
                syn_att = syn_att.cuda()
                syn_noise = syn_noise.cuda()                
            for i in range(nclass):
                iclass = classes[i]          
                if opt.mergeMapping:
                    iclass_att = attribute[iclass].cuda()
                    clip_encoder = glob_model.clip_encoder
                    clip_att = clip_encoder([iclass],torch.unsqueeze(iclass_att,0))
                    # clip_att = clip_encoder([iclass],torch.unsqueeze(iclass_att,0))
                    clip_att = clip_att.repeat(num, 1)
                    syn_att.copy_(clip_att)
                else:
                    iclass_att = attribute[iclass]
                    syn_att.copy_(iclass_att.repeat(num, 1))
                syn_noise.normal_(0, 1)
                output = netG(Variable(syn_noise, volatile=True), Variable(syn_att, volatile=True))
                syn_feature.narrow(0, i*num, num).copy_(output.data.cpu())
                syn_label.narrow(0, i*num, num).fill_(iclass)
            return syn_feature, syn_label



# local client model training and current model evaluation
class LocalUpdateLM(object):
    def __init__(self, args, data_loader, nround):
        self.args = args
        self.round = nround

        self.data = data_loader
        # self.data_loader = DataLoader(DatasetSplitLM(dataset, list(idxs)), batch_size=self.args.batchsize, shuffle=True, num_workers=4)
        self.cls_criterion = nn.NLLLoss().cuda()


    # def sample(self):
    #     batch_feature, batch_label, batch_att = self.data.next_batch(self.args.batch_size)
    #     self.input_res.copy_(batch_feature)
    #     self.input_att.copy_(batch_att)
    #     self.input_label.copy_(util.map_label(batch_label, self.data.seenclasses))
    #     return self.input_res,self.input_att,self.input_label


    # updating local model parameters
    def update_weights(self, model, cur_epoch, idx_client):
        opt=self.args
        # mapping network parameters
        # when local epoch > 1, the copy server model should also be updated shared the merit of mutual learning

        # ignored_params = list(map(id, model.Ext_Cls.model.fc.parameters() )) + list(map(id, model.Ext_Cls.classifier.parameters() ))
     

        # feature embedding network parameters
        # shitong filter: first parameter: return True/False; filter will choose from the second parameter which satisfy the first
        # base_params = filter(lambda p: id(p) not in ignored_params, model.Ext_Cls.parameters())
        input_res = torch.FloatTensor(opt.batch_size, opt.resSize).cuda()
        input_att = torch.FloatTensor(opt.batch_size, opt.attSize).cuda()
        noise = torch.FloatTensor(opt.batch_size, opt.nz).cuda()

        one = torch.tensor(1, dtype=torch.float).cuda()
        mone = one*-1
        input_label = torch.LongTensor(opt.batch_size).cuda()
        input_unmap_label = torch.LongTensor(opt.batch_size)

        def sample():
            batch_feature, batch_label, batch_att = data.next_batch(opt.batch_size)
            input_res.copy_(batch_feature)
            input_att.copy_(batch_att)
            input_unmap_label.copy_(batch_label)

            input_label.copy_(util.map_label(batch_label, data.seenclasses))

        
        if cur_epoch < 15:
            decay_factor = 1.0
        elif cur_epoch<30:
            decay_factor = 0.1
        elif cur_epoch<45:
            decay_factor = 0.01
        else: 
            decay_factor = 0.001
    #    #shitong
        decay_factor=1

        data = self.data

        optimizerD = optim.Adam(model.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        optimizerG = optim.Adam([{'params':model.netG.parameters(),'lr': opt.lr*decay_factor},
                         {'params':model.clip_encoder.mapping.parameters(),'lr':opt.lr*10*decay_factor}], betas=(opt.beta1, 0.999))


        # local LR scheduler
        # scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        # train and update
        #average meter

        #from xian cvpr 18               


        pretrain_cls = classifier.CLASSIFIER(data.train_feature, util.map_label(data.train_label, data.seenclasses), data.seenclasses.size(0), opt.resSize, opt.cuda, 0.001, 0.5, 50, 100, opt.pretrain_classifier)

        for p in pretrain_cls.model.parameters(): # set requires_grad to False
            p.requires_grad = False

        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        loss_meter = AverageMeter('Total loss', ':6.3f')

        for epoch in range(opt.local_ep):
            FP = 0 
            mean_lossD = 0
            mean_lossG = 0
            for i in range(0, data.ntrain, opt.batch_size):
                ############################
                # (1) Update D network: optimize WGAN-GP objective, Equation (2)
                ###########################
                for p in model.netD.parameters(): # reset requires_grad
                    p.requires_grad = True # they are set to False below in netG update

                for iter_d in range(opt.critic_iter):
                    sample()
                    model.netD.zero_grad()
                    # train with realG
                    # sample a mini-batch
                    sparse_real = opt.resSize - input_res[1].gt(0).sum()
                    input_resv = Variable(input_res)
                    input_attv = Variable(input_att)

                    criticD_real = model.netD(input_resv, input_attv)
                    criticD_real = criticD_real.mean()
                    criticD_real.backward(mone)

                    # train with fakeG
                    noise.normal_(0, 1)
                    noisev = Variable(noise)
                    
                    clip_att = model.clip_encoder(input_label,input_attv)
                    fake = model.netG(noisev, clip_att)

                    fake_norm = fake.data[0].norm()
                    sparse_fake = fake.data[0].eq(0).sum()
                    
                    criticD_fake =  model.netD(fake.detach(), input_attv)
                    criticD_fake = criticD_fake.mean()
                    criticD_fake.backward(one)

                    # gradient penalty
                    gradient_penalty = calc_gradient_penalty( opt,model.netD, input_res, fake.data, input_att)
                    gradient_penalty.backward()

                    Wasserstein_D = criticD_real - criticD_fake
                    D_cost = criticD_fake - criticD_real + gradient_penalty
                    optimizerD.step()

                ############################
                # (2) Update G network: optimize WGAN-GP objective, Equation (2)
                ###########################
                for p in  model.netD.parameters(): # reset requires_grad
                    p.requires_grad = False # avoid computation

                optimizerG.zero_grad()

                input_attv = Variable(input_att)
                noise.normal_(0, 1)
                noisev = Variable(noise)

                clip_att = model.clip_encoder(input_label,input_attv)


                fake =  model.netG(noisev, clip_att)
                criticG_fake =  model.netD(fake, input_attv)
                criticG_fake = criticG_fake.mean()
                G_cost = -criticG_fake
                # classification loss
                c_errG = self.cls_criterion(pretrain_cls.model(fake), Variable(input_label))
                errG = G_cost + opt.cls_weight*c_errG
                errG.backward()
                optimizerG.step()

            mean_lossG /=  data.ntrain / opt.batch_size 
            mean_lossD /=  data.ntrain / opt.batch_size 
            print('[%d/%d] Loss_D: %.4f Loss_G: %.4f, Wasserstein_dist: %.4f, c_errG:%.4f'
                    % (epoch, opt.local_ep, D_cost.data.item(), G_cost.data.item(), Wasserstein_D.data.item(), c_errG.data.item()))

            # evaluate the model, set G to evaluation mode
            model.netG.eval()
            # Generalized zero-shot learning

            if opt.gzsl:
                syn_feature, syn_label = generate_syn_feature(model,data.unseenclasses, data.attribute, opt.syn_num,opt)
                train_X = torch.cat((data.train_feature, syn_feature), 0)
                train_Y = torch.cat((data.train_label, syn_label), 0)
                nclass = opt.nclass_all
                cls = classifier2.CLASSIFIER(train_X, train_Y, data, nclass, opt.cuda, opt.classifier_lr, 0.5, 25, opt.syn_num, True)
                print('unseen=%.4f, seen=%.4f, h=%.4f' % (cls.acc_unseen, cls.acc_seen, cls.H))
            # Zero-shot learning
            else:
                syn_feature, syn_label = generate_syn_feature(model, data.unseenclasses, data.attribute, opt.syn_num,opt) 
                cls = classifier2.CLASSIFIER(syn_feature, util.map_label(syn_label, data.unseenclasses), data, data.unseenclasses.size(0), opt.cuda, opt.classifier_lr, 0.5, 25, opt.syn_num, False)
                acc = cls.acc
                print('unseen class accuracy= ', acc)
                
            # reset G to training mode
            model.netG.train()
        if opt.gzsl:
            return{
                'model_params':model.state_dict(),
                'acc_unseen':cls.acc_unseen,
                'acc_seen':cls.acc_seen,
                'H':cls.H
            }
        else:
            return{
                'model_params':model.state_dict(),
                'acc_unseen':acc
            }      

        
    # evaluating current model
    def evaluate(self, data_loader, model, idx_client=0,disEN=False):
        model = model.cuda()       
        model.train(False)  # Set model to evaluate mode

        running_loss = 0.0
        running_corrects = 0.0

        # Iterate over data.
        for data in data_loader:
            # get the inputs
            inputs, labels = data
            now_batch_size,c,h,w = inputs.shape

           
            # wrap data in Variable           
            inputs = Variable(inputs.cuda())
            labels = Variable(labels.cuda())
            
            outputs = model(inputs, idx_client)
            # compute accuracy and loss of current batch

            _, preds = torch.max(outputs.data, 1)
            loss = self.criterion_ce(outputs, labels)

            running_loss += loss.item() * now_batch_size
            running_corrects += float(torch.sum(preds == labels.data))

        # compute accuracy and loss of current epoch
        epoch_loss = running_loss / (len(data_loader)*self.args.batchsize)
        epoch_acc = running_corrects / (len(data_loader)*self.args.batchsize)


        # return evaluation loss and accuracy
        return epoch_loss, epoch_acc

