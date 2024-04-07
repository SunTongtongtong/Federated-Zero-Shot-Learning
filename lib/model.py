# 
# FedReID model: feature embedding network + mapping network
#

import torch.nn as nn
from torch.nn import init
from torchvision import models
from torch.autograd import Variable
from torch.nn import functional as F
import torch
from .resnet_cbam import resnet50_cbam

import cvpr18xian.model as model
from CoOp.trainers.cocoop import CoCoOp
# from dassl.engine import build_trainer
from CoOp.trainers.cocoop import *
import numpy as np

from attribute_name import att_name




# Weight initialisation
def weights_init_kaiming(m):
    classname = m.__class__.__name__

    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)

# Multi-Layer Perceptron (MLP) as a mapping network
class MLP(nn.Module):
    def __init__(self, input_dim, class_num, dropout=True, relu=True, num_bottleneck=512):
        super(MLP, self).__init__()

        add_block = []
        add_block += [nn.Linear(input_dim, num_bottleneck)] 
        add_block += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if dropout:
            add_block += [nn.Dropout(p=0.5)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier

    def forward(self, x):
        #print(x.shape)
        x = self.add_block(x)
        x = self.classifier(x)
        return x

    @staticmethod
    def _init_fc(fc):
        nn.init.kaiming_normal_(fc.weight, mode='fan_out')
        #nn.init.normal_(fc.weight, std=0.001)
        nn.init.constant_(fc.bias, 0.)

# Feature embedding network 
class embedding_net(nn.Module):
    def __init__(self, num_ids_client, feat_dim=2048,AN=False):
        super(embedding_net, self).__init__()
        model_backbone = models.resnet50(pretrained=True)
        model_backbone.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.model = model_backbone
        self.classifier =  MLP(feat_dim, num_ids_client)

    def forward(self, x, repre=False):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        representation = x.view(x.size(0), x.size(1))
        x = self.classifier(representation)
        if repre:
            return representation,x
        else:
            return x

# Feature embedding network for testing
class embedding_net_test(nn.Module):
    def __init__(self, model):
        super(embedding_net_test, self).__init__()
        self.model = model.model

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = x.view(x.size(0), x.size(1)) # embedding feature representation
        return x

class Discriminator(nn.Module):
    def __init__(self,opt):
        super(Discriminator, self).__init__()

        # self.label_embedding = nn.Embedding(opt.n_classes, opt.n_classes)
        self.model = nn.Sequential(
            # nn.Linear(opt.n_classes + int(np.prod(img_shape)), 512),
            nn.Linear(opt.attr_num+2048, 512),  # shitong: 2048 is the feature length of the extractor output
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, representation,labels):
        # Concatenate label embedding and image to produce input
 
        # d_in = torch.cat((img.view(img.size(0), -1), self.label_embedding(labels)), -1)
        d_in = torch.cat((labels.view(labels.size(0),-1),representation),-1)
        validity = self.model(d_in)
        return validity

def block(in_feat, out_feat, normalize=True):
    layers = [nn.Linear(in_feat, out_feat)]
    if normalize:
        layers.append(nn.BatchNorm1d(out_feat, 0.8))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return layers


class Generator(nn.Module):
    def __init__(self,opt):
        # add one noise to each attribute currently
        super(Generator, self).__init__()

        # self.label_emb = nn.Embedding(opt.n_classes, opt.n_classes) # shitong, make it easier

        self.model = nn.Sequential(
            *block(opt.latent_dim + opt.attr_num, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 2048),
            nn.Tanh()  # shitong: 
        )
        self.model.apply(weights_init_kaiming)
    def forward(self, noise, labels):
        # labels shape [Attr_NUM, Attr_option]
        # gen_input = torch.cat((self.label_emb(labels), noise), -1)

        #noise is a matrix also, each attribute with a noise
        gen_input = torch.cat((labels,noise),-1)
        output = self.model(gen_input.reshape(noise.size()[0],-1))
        return output


class Encoder(nn.Module):
    def __init__(self,opt):
        super(Encoder,self).__init__()
        self.model = nn.Sequential(
            *block(2048, 1024),
            *block(1024, 512),
            *block(512, 256),
            *block(256, opt.attr_num),
        )
        self.model.apply(weights_init_kaiming)

    def forward(self,x):
        x = self.model(x)
        return x

class Decoder(nn.Module):
    def __init__(self,opt):
        super(Decoder,self).__init__()
        self.model = nn.Sequential(
            *block(opt.attr_num, 256),
            *block(256, 512),
            *block(512, 1024),
            *block(1024, 2048),
        )
        self.model.apply(weights_init_kaiming)
        
    def forward(self):
        x = self.model(x)
        return


class clip_input_model(nn.Module):
    def __init__(self,data_loader,opt):
        super(clip_input_model, self).__init__()
        self.model, preprocess = clip.load('ViT-B/16', 'cuda')
        # self.all_classnames = [global_loader.all_classnames[label] for label in torch.unique(global_loader.label)]
        self.classnames = data_loader.classnames

        print("Turning off gradients in both the image and the text encoder")
        
        for name, param in self.model.named_parameters():
            param.requires_grad_(False)
        # self.att_name = att_name[opt.dataset]        
        # text_inputs = torch.cat([clip.tokenize(f"a photo of an animal with {c}") for c in attribute_names]).to('cuda')
        self.global_loader = data_loader
        self.unseenclassnames = data_loader.unseenclassnames
        self.opt = opt
        self.ntrain = data_loader.ntrain
        self.batch_size = 64
        # if self.opt.clip_att:
        #     self.binary_attribute = self.global_loader.binary_attribute # shape [50,85]

    def forward(self,batch_label,gt_attribute,test=False,GZSL=False):        
        # if test:
        #     classname = [self.unseenclassnames[label] for label in batch_label] # batch_label should before map unique 
        # else:
        #     classname = [self.classnames[label] for label in batch_label] # batch_label should before map unique 
        # text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}.") for c in classname]).to('cuda')
        # with torch.no_grad():
        #     text_features = self.model.encode_text(text_inputs).cuda()
        # text_features =  text_features/text_features.norm(dim=-1, keepdim=True)
        # return text_features
        # random_att_idx = [np.random.choice(np.where(a>0.1)[0]) for a in gt_attribute.cpu().numpy()]
        # attribute_name = [self.att_name[idx] for idx in random_att_idx]
        if test:
            if GZSL:
                classname = [self.global_loader.seen_unseen_classnames[label] for label in batch_label]
            else:
            # classname = [self.unseenclassnames[label] for label in batch_label] # batch_label should before map unique 
            # generate both seen and unseen for FL
                classname = [self.unseenclassnames[label] for label in batch_label] 
        else:
            classname = [self.classnames[label] for label in batch_label] # batch_label should before map unique 
        # text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}{a}.") for c,a in zip(classname,attribute_name)]).to('cuda')
        # text_inputs = torch.cat([clip.tokenize(f"a photo of a bird {a}.") for a in attribute_name]).to('cuda')
        # if self.opt.dataset=='CUB':
        #     text_inputs = torch.cat([clip.tokenize(f"a photo of a bird.") for a in attribute_name]).to('cuda')
        # elif self.opt.dataset=='SUN':
        #     text_inputs = torch.cat([clip.tokenize(f"a photo of a scene.") for a in attribute_name]).to('cuda')
        # else:
        text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}.") for c in classname]).to('cuda')
      
        # text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}.") for c in classname]).to('cuda')
        with torch.no_grad():
            text_features = self.model.encode_text(text_inputs).cuda()
        text_features =  text_features/text_features.norm(dim=-1, keepdim=True)
        return text_features


class clip_encoder(nn.Module):
    def __init__(self,data_loader,opt):
        super(clip_encoder,self).__init__()
        if opt.learnable_clip:
            self.clip = learnable_CLIP(data_loader,opt)
        else:
            self.clip = clip_input_model(data_loader,opt)

        self.mapping = model.MLP_compressCLIP(opt)
        self.opt = opt
        print('!!!!!!!!noise 0.1 ')


    def forward(self,batch_label,gt_attribute,test=False,GZSL=False):
        text_features = self.clip(batch_label,gt_attribute,test,GZSL).float()

        noise = torch.FloatTensor( batch_label.__len__(), 512)
        noise.normal_(0, 0.1)
        noisev = Variable(noise).cuda()
        text_features = text_features+noisev

        attribute_features = torch.cat([text_features,gt_attribute],dim=-1)
        # x = self.mapping(attribute_features)

        return attribute_features



class attribute_align_net(nn.Module):
    def __init__(self,opt,dataloader):
        super().__init__()
        self.netD = model.MLP_CRITIC(opt)
        self.netG  = model.MLP_G(opt)
        self.clip_encoder = clip_encoder(dataloader,opt)   

class attribute_align_net_test(nn.Module):
    def __init__(self,opt,dataloader):
        super().__init__()
        self.netG  = model.MLP_G(opt)
        if opt.mergeMapping:
            self.clip_encoder = clip_encoder(dataloader,opt)   
        # self.netD = model.MLP_CRITIC(opt)


    