#
# Extract Feature Representation of Testing Data
#

from __future__ import print_function, division
import torch
import scipy.io


from utils.get_dataset import get_dataset
from utils.load_network import load_network, load_network_mergeImageNet

from config_test import opt
from lib.model import embedding_net, embedding_net_test

from evaluation.get_id import get_id, get_id_cuhk_msmt
from evaluation.eval_feat_ext import eval_feat_ext,eval_feat_ext_joint#, fliplr
from lib.model import embedding_net

#shitong
import torch.optim as optim

from evaluate import evaluate
# from adaBN import bn_update
# from tent import tent

import copy

str_ids = opt.gpu_ids.split(',')
gpu_ids = []
for str_id in str_ids:
    id = int(str_id)
    if id >=0:
        gpu_ids.append(id)

# Set gpu ids
if len(gpu_ids)>0:
    torch.cuda.set_device(gpu_ids[0])

use_gpu = torch.cuda.is_available()

def main(opt):
    # Set GPU  
    # Load testing data
    print('----------Load testing data----------')
    name = opt.test_data_dir.split('/')[6]
    if opt.test_data_dir.split('/')[5]=='10_split_targetDataset':
        print('on 10 splits small datasets')
        rank1_list = []
        mAP_list = []
        for i in range(1,11):
            old = str(i-1)

            opt.test_data_dir = opt.test_data_dir.replace('split-'+str(old),'split-'+str(i))
            feat_extract(name)
            rank1,mAP = evaluate()
            rank1_list.append(rank1)
            mAP_list.append(mAP)
        print('*********Result on '+name+'*********')
        print('10 splits of result on dataset {} is Rank1: {}, mAP {}'.format(name,sum(rank1_list)/len(rank1_list),sum(mAP_list)/len(mAP_list))) 
    else:
        feat_extract(name)
        rank1,mAP = evaluate()
        print('*********Result on '+name+'*********')
        print('Large datasets on {} is Rank1: {}, mAP {}'.format(name,rank1,mAP) )

 
def feat_extract(name):
    image_datasets, dataloaders  = get_dataset(opt, is_training=False) #change opt test dir
    print('Done.')

    # Get camera and identity labels of gallery and query
    gallery_path = image_datasets['gallery'].imgs
    query_path = image_datasets['query'].imgs

    # gallery_cam, gallery_label = get_id(gallery_path)
    # query_cam, query_label = get_id(query_path)

    if gallery_path[0][0].split('/')[-5] not in ['msmt17', 'cuhk03-np']:
        gallery_cam, gallery_label = get_id(gallery_path)
        query_cam, query_label = get_id(query_path)
    else:
        gallery_cam, gallery_label = get_id_cuhk_msmt(gallery_path)
        query_cam, query_label = get_id_cuhk_msmt(query_path)

    print('----------Extracting features----------')
    # Model initialisation, we use four local client in current version (Duke, Market, MSMT, CUHK03)
    # model = embedding_net([702, 751, 1041, 767])
    print(name)
    num_class_dict = {'dukemtmc-reid':702,'market1501':751,'msmt17':1041,'cuhk03-np':767}
    if name in ['dukemtmc-reid','market1501','msmt17','cuhk03-np']:
        model = embedding_net(num_class_dict[name],AN = opt.AN)
        model = embedding_net_test(model)

        model = load_network(model, opt.model_name, gpu_ids,name) # Model restoration from saved model
        # model = embedding_net_test(model)

    else:
        model = embedding_net(3261,AN=opt.AN)
        model = embedding_net_test(model)
       # model = load_network_mergeImageNet(model, opt.model_name, gpu_ids,name) # Model restoration from saved model
        model = load_network(model, opt.model_name, gpu_ids,name) # Model restoration from saved model

    #####################joint features as cat##########################
    # models = []
    # for name in num_class_dict.keys():

    #     model = embedding_net(num_class_dict[name],AN = opt.AN)
    #     model = load_network(model, opt.model_name, gpu_ids,name)
    #     model = embedding_net_test(model)
    #     model = model.cuda()
    #     model = model.eval()
    

    #     models.append(model)
    #####################joint features as cat##########################


    # Remove the mapping network and set to embedding feature extraction
    model = model.cuda()    
    # Change to test modei

    # bn_update(model, dataloaders['gallery'])#,cumulative = not args.adabn_emv)

    model = model.eval()

    # model = tent.configure_model(model)
    # params, param_names = tent.collect_params(model)
    # optimizer = optim.Adam(params, lr=1e-3,betas=(0.9, 0.999),weight_decay=0.0)
    # tented_model = tent.Tent(model, optimizer)
    # model = tented_model

     
 # Extract feature for joint concat features
    # gallery_feature = eval_feat_ext_joint(models, dataloaders['gallery'])
    # print('Done gallery.')
    # query_feature = eval_feat_ext_joint(models, dataloaders['query'])
    # print('Done query.')


    # Extract feature

    model.is_query = False
    gallery_feature = eval_feat_ext(model, dataloaders['gallery'])
    print('Done gallery.')
    model.is_query = True
    query_feature = eval_feat_ext(model, dataloaders['query'])
    print('Done query.')

    print('----------Saving features----------')
    result = {'gallery_f':gallery_feature.numpy(),'gallery_label':gallery_label,'gallery_cam':gallery_cam,
              'query_f':query_feature.numpy(),'query_label':query_label,'query_cam':query_cam, 'gallery_path':gallery_path,'query_path':query_path}
    scipy.io.savemat('result_feature.mat',result)
    print('Done.')



if __name__ == '__main__':
    main(opt)


