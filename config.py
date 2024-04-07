#
# Model training parameter setting
#

import argparse
import os

######################################################################
# Options
# --------
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids', default='0', type=str,help='GPU IDs: e.g. 0, 1, 2, ...')
parser.add_argument('--name',default='debug', type=str, help='Model Name')
parser.add_argument('--logs_dir', type=str, help='path of logs',
    default=os.path.join(os.path.dirname(os.path.abspath(__file__)),'model_save'))

# Four local clients
parser.add_argument('--nusers', default=4, type=int, help='number of clients in federated learning')


# Hyper-parameters
#for testing
parser.add_argument('--batchsize', default=32, type=int, help='batchsize')
# parser.add_argument('--lr_init', default=0.01, type=float, help='initial learning rate')
parser.add_argument('--frac', type=float, default=1.0, help='the fraction of clients: C')
parser.add_argument('--agg', type=str, default='avg', help='Federated average strategy')
parser.add_argument('--dp', type=float, default=0.000, help='differential privacy: beta')
parser.add_argument('--local_bs', type=int, default=32, help="local batch size")
parser.add_argument('--local_ep', type=int, default=1, help="number of local epochs: t_max")
parser.add_argument('--global_ep', type=int, default=100, help="number of global epochs: k_max")
parser.add_argument('--T', type=int, default=3, help="temperature to control the softness of probability distributions")
parser.add_argument('--alpha_mu', type=float, default=0.5, help="update momentum in local weight aggregation: alpha")

# shitong, for attribtue alignment
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space; length of the noise")

#change later, get this two from saved attribute label
parser.add_argument("--attr_num", type=int, default=39)
# parser.add_argument("--attr_opt", type=int, default=6)


# for GZSL
parser.add_argument('--dataset', default='FLO', help='FLO')
parser.add_argument('--dataroot', default='/import/sgg-homes/ss014/datasets/ZSL/xlsa17/data/', help='path to dataset')
parser.add_argument('--matdataset', default=True, help='Data in matlab format')
parser.add_argument('--image_embedding', default='res101')
parser.add_argument('--class_embedding', default='att')
parser.add_argument('--syn_num', type=int, default=100, help='number features to generate per class')
parser.add_argument('--gzsl', action='store_true', default=False, help='enable generalized zero-shot learning')
parser.add_argument('--preprocessing', action='store_true', default=False, help='enbale MinMaxScaler on visual features')
parser.add_argument('--standardization', action='store_true', default=False)
parser.add_argument('--validation', action='store_true', default=False, help='enable cross validation mode')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
parser.add_argument('--resSize', type=int, default=2048, help='size of visual features')
parser.add_argument('--attSize', type=int, default=1024, help='size of semantic features')
parser.add_argument('--nz', type=int, default=312, help='size of the latent z vector')
parser.add_argument('--ngh', type=int, default=4096, help='size of the hidden units in generator')
parser.add_argument('--ndh', type=int, default=1024, help='size of the hidden units in discriminator')
# parser.add_argument('--nepoch', type=int, default=2000, help='number of epochs to train for')
parser.add_argument('--critic_iter', type=int, default=5, help='critic iteration, following WGAN-GP')
parser.add_argument('--lambda1', type=float, default=10, help='gradient penalty regularizer, following WGAN-GP')
parser.add_argument('--cls_weight', type=float, default=1, help='weight of the classification loss')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate to train GANs ')
parser.add_argument('--classifier_lr', type=float, default=0.001, help='learning rate to train softmax classifier')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', default=False, help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--pretrain_classifier', default='', help="path to pretrain classifier (to continue training)")
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--netG_name', default='')
parser.add_argument('--netD_name', default='')
parser.add_argument('--outf', default='./checkpoint/', help='folder to output data and model checkpoints')
parser.add_argument('--outname', help='folder to output data and model checkpoints')
parser.add_argument('--save_every', type=int, default=100)
parser.add_argument('--print_every', type=int, default=1)
parser.add_argument('--val_every', type=int, default=10)
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--nclass_all', type=int, default=200, help='number of all classes')

parser.add_argument('--learnable_clip', action='store_true', default=False, help='enables cuda')
parser.add_argument('--clip_att', action='store_true', default=False, help='enables cuda')
parser.add_argument('--clip_noise', action='store_true', default=False, help='enables cuda')
parser.add_argument('--mergeMapping', action='store_true', default=True, help='enables cuda')

opt = parser.parse_args()
