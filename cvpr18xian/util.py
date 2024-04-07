import h5py
import numpy as np
import scipy.io as sio
from sklearn.manifold import TSNE
import torch
from sklearn import preprocessing
import sys
import os.path as osp
from torchvision import datasets,transforms
from PIL import Image
from os.path import exists
# from tSNE import visualize_tSNE
import torch

from torch.utils.tensorboard import SummaryWriter



def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def map_label(label, classes):
    mapped_label = torch.LongTensor(label.size())
    for i in range(classes.size(0)):
        mapped_label[label==classes[i]] = i    

    return mapped_label


class Logger(object):
    def __init__(self, filename):
        self.filename = filename
        f = open(self.filename+'.log', "a")
        f.close()

    def write(self, message):
        f = open(self.filename+'.log', "a")
        f.write(message)  
        f.close()

class DATA_LOADER(object):
    def __init__(self, opt):
        self.data_transforms = transforms.Compose([transforms.RandomResizedCrop(224,interpolation=3),
                               transforms.RandomHorizontalFlip(p=0.5),
                               transforms.ToTensor(),
                               transforms.Normalize([0.48145466, 0.4578275, 0.40821073],[0.26862954, 0.26130258, 0.27577711])         # value from coop --clip                      
                               ])

        if opt.matdataset:
            if opt.dataset == 'imageNet1K':
                self.read_matimagenet(opt)
            else:
                # if opt.clip_image:
                self.read_matdataset_clip(opt)
                # else:
                # self.read_matdataset(opt)
        self.index_in_epoch = 0
        self.epochs_completed = 0
        
        self.name2label = {name:id for id,name in enumerate(self.all_classnames)}  # the sequence in all_classnames is the correct label

        # shitong add here, provide raw images to extract features further

        
        #shitong
        if opt.dataset=='AWA2':            
            # self.rawdatasets = datasets.ImageFolder('/import/sgg-homes/ss014/datasets/Animals_with_Attributes2/JPEGImages',self.data_transforms)
            # self.rawloader = torch.utils.data.DataLoader(self.rawdatasets,batch_size=opt.batchsize,shuffle=True)
            # self.name2label_raw = self.rawdatasets.class_to_idx  # the sequence in all_classnames is the correct label
            # self.rawLabel2label = {self.rawdatasets.class_to_idx[key]:self.name2label[key] for key in self.rawdatasets.class_to_idx.keys()}  # map from imagefolder label to xlra/data/label
            
            self.all_classnames = [name.replace('+',' ') for name in self.all_classnames]
            # self.train_imgs_path =[]
            # self.train_labels_clip = []

            # for img in self.rawdatasets.imgs:
            #     img_path,raw_label = img
            #     label = self.rawLabel2label[raw_label]
            #     if label in self.train_label:
            #         self.train_labels_clip.append(label)
                


    # not tested
    
    def read_h5dataset(self, opt):
        # read image feature
        fid = h5py.File(opt.dataroot + "/" + opt.dataset + "/" + opt.image_embedding + ".hdf5", 'r')
        feature = fid['feature'][()]
        label = fid['label'][()] 
        trainval_loc = fid['trainval_loc'][()] 
        train_loc = fid['train_loc'][()] 
        val_unseen_loc = fid['val_unseen_loc'][()] 
        test_seen_loc = fid['test_seen_loc'][()] 
        test_unseen_loc = fid['test_unseen_loc'][()] 
        fid.close()
        # read attributes
        fid = h5py.File(opt.dataroot + "/" + opt.dataset + "/" + opt.class_embedding + ".hdf5", 'r')
        self.attribute = fid['attribute'][()]
        fid.close()

        if not opt.validation:
            self.train_feature = feature[trainval_loc] 
            self.train_label = label[trainval_loc] 
            self.test_unseen_feature = feature[test_unseen_loc] 
            self.test_unseen_label = label[test_unseen_loc] 
            self.test_seen_feature = feature[test_seen_loc] 
            self.test_seen_label = label[test_seen_loc] 
        else:
            self.train_feature = feature[train_loc] 
            self.train_label = label[train_loc] 
            self.test_unseen_feature = feature[val_unseen_loc] 
            self.test_unseen_label = label[val_unseen_loc] 

        self.seenclasses = np.unique(self.train_label)
        self.unseenclasses = np.unique(self.test_unseen_label)
        self.nclasses = self.seenclasses.size(0)

    def read_matimagenet(self, opt):
        if opt.preprocessing:
            print('MinMaxScaler...')
            scaler = preprocessing.MinMaxScaler()
            matcontent = h5py.File(opt.dataroot + "/" + opt.dataset + "/" + opt.image_embedding + ".mat", 'r')
            feature = scaler.fit_transform(np.array(matcontent['features']))
            label = np.array(matcontent['labels']).astype(int).squeeze() - 1
            feature_val = scaler.transform(np.array(matcontent['features_val']))
            label_val = np.array(matcontent['labels_val']).astype(int).squeeze() - 1
            matcontent.close()
            matcontent = h5py.File('/BS/xian/work/data/imageNet21K/extract_res/res101_1crop_2hops_t.mat', 'r')
            feature_unseen = scaler.transform(np.array(matcontent['features']))
            label_unseen = np.array(matcontent['labels']).astype(int).squeeze() - 1
            matcontent.close()
        else:
            matcontent = h5py.File(opt.dataroot + "/" + opt.dataset + "/" + opt.image_embedding + ".mat", 'r')
            feature = np.array(matcontent['features'])
            label = np.array(matcontent['labels']).astype(int).squeeze() - 1
            feature_val = np.array(matcontent['features_val'])
            label_val = np.array(matcontent['labels_val']).astype(int).squeeze() - 1
            matcontent.close()


        matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.class_embedding + ".mat")
        self.attribute = torch.from_numpy(matcontent['w2v']).float()
        self.train_feature = torch.from_numpy(feature).float()
        self.train_label = torch.from_numpy(label).long() 
        self.test_seen_feature = torch.from_numpy(feature_val).float()
        self.test_seen_label = torch.from_numpy(label_val).long() 
        self.test_unseen_feature = torch.from_numpy(feature_unseen).float()
        self.test_unseen_label = torch.from_numpy(label_unseen).long() 
        self.ntrain = self.train_feature.size()[0]
        self.seenclasses = torch.from_numpy(np.unique(self.train_label.numpy()))
        self.unseenclasses = torch.from_numpy(np.unique(self.test_unseen_label.numpy()))
        self.train_class = torch.from_numpy(np.unique(self.train_label.numpy()))
        self.ntrain_class = self.seenclasses.size(0)

        self.ntest_class = self.unseenclasses.size(0)


    def read_matdataset(self, opt):
   
        mat_file_path = osp.join(osp.join(opt.dataroot,opt.dataset),opt.image_embedding+'.mat')
        matcontent = sio.loadmat(mat_file_path)
        feature = matcontent['features'].T
        label = matcontent['labels'].astype(int).squeeze() - 1
        #shitong
        image_paths = matcontent['image_files'].squeeze()
        self.image_paths = [path.tolist()[0].split('//')[1] for path in image_paths]
        dataset_root = '/import/sgg-homes/ss014/datasets/Animals_with_Attributes2'
        self.image_paths = [osp.join(dataset_root,path) for path in self.image_paths]

        matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.class_embedding + "_splits.mat")
        # numpy array index starts from 0, matlab starts from 1

        trainval_loc = matcontent['trainval_loc'].squeeze() - 1
        train_loc = matcontent['train_loc'].squeeze() - 1
        val_unseen_loc = matcontent['val_loc'].squeeze() - 1
        test_seen_loc = matcontent['test_seen_loc'].squeeze() - 1
        test_unseen_loc = matcontent['test_unseen_loc'].squeeze() - 1
        
        class_names = matcontent['allclasses_names']
        # class_names = [name[0][0].tolist() for name in class_names]

        if opt.dataset=='CUB':
            print('CUB datasets name correct, remove number')
            class_names = [name[0][0].tolist().split('.')[1].replace('_',' ') for name in class_names]        
        else:
            class_names = [name[0][0].tolist() for name in class_names]



        self.attribute = torch.from_numpy(matcontent['att'].T).float() 
        if not opt.validation:
            if opt.preprocessing:
                if opt.standardization:
                    print('standardization...')
                    scaler = preprocessing.StandardScaler()
                else:
                    scaler = preprocessing.MinMaxScaler()
                
                _train_feature = scaler.fit_transform(feature[trainval_loc])
                _test_seen_feature = scaler.transform(feature[test_seen_loc])
                _test_unseen_feature = scaler.transform(feature[test_unseen_loc])
                self.train_feature = torch.from_numpy(_train_feature).float()
                mx = self.train_feature.max()
                self.train_feature.mul_(1/mx)   # shape 19832,2048
                self.train_label = torch.from_numpy(label[trainval_loc]).long() 
                self.test_unseen_feature = torch.from_numpy(_test_unseen_feature).float()
                self.test_unseen_feature.mul_(1/mx)
                self.test_unseen_label = torch.from_numpy(label[test_unseen_loc]).long() 
                self.test_seen_feature = torch.from_numpy(_test_seen_feature).float() 
                self.test_seen_feature.mul_(1/mx)
                self.test_seen_label = torch.from_numpy(label[test_seen_loc]).long()
            else:
                self.train_feature = torch.from_numpy(feature[trainval_loc]).float()
                self.train_label = torch.from_numpy(label[trainval_loc]).long() 
                self.test_unseen_feature = torch.from_numpy(feature[test_unseen_loc]).float()
                self.test_unseen_label = torch.from_numpy(label[test_unseen_loc]).long() 
                self.test_seen_feature = torch.from_numpy(feature[test_seen_loc]).float() 
                self.test_seen_label = torch.from_numpy(label[test_seen_loc]).long()
        else:
            self.train_feature = torch.from_numpy(feature[train_loc]).float()
            self.train_label = torch.from_numpy(label[train_loc]).long()
            self.test_unseen_feature = torch.from_numpy(feature[val_unseen_loc]).float()
            self.test_unseen_label = torch.from_numpy(label[val_unseen_loc]).long() 
    
        self.seenclasses = torch.from_numpy(np.unique(self.train_label.numpy()))
        self.unseenclasses = torch.from_numpy(np.unique(self.test_unseen_label.numpy()))
        self.ntrain = self.train_feature.size()[0]
        self.ntrain_class = self.seenclasses.size(0)
        self.ntest_class = self.unseenclasses.size(0)
        self.train_class = self.seenclasses.clone()
        self.allclasses = torch.arange(0, self.ntrain_class+self.ntest_class).long()


                
        self.train_mapped_label = map_label(self.train_label, self.seenclasses)  # change label from 绝对label 到seen class里的相对label

        # shitong
        self.all_classnames = class_names



    def read_matdataset_clip(self, opt):
   
        mat_file_path = osp.join(osp.join(opt.dataroot,opt.dataset),opt.image_embedding+'.mat')
        matcontent = sio.loadmat(mat_file_path)
        label = matcontent['labels'].astype(int).squeeze() - 1
        #shitong
        image_paths = matcontent['image_files'].squeeze()
        # if opt.dataset=='AWA2':
        #     self.image_paths = [path.tolist()[0].split('//')[1] for path in image_paths]
        #     self.image_paths = [osp.join(opt.dataset_root,path) for path in self.image_paths]


        # if opt.dataset=='SUN':
        #     self.image_paths = [path.tolist()[0].split('SUN')[1] for path in image_paths]
        #     self.image_paths = [osp.join(opt.dataset_root,path[1:]) for path in self.image_paths]

        # dataset_root = '/import/sgg-homes/ss014/datasets/Animals_with_Attributes2'
        feature = matcontent['features'].T

      
        # here for CLIP image features 
        # if exists('clip_img_feature_'+opt.dataset+'.npy'):
        #     print('####### Extracting feature from file clip_img_feature.npy #########')
        #     with open('clip_img_feature_'+opt.dataset+'.npy','rb') as f:               
        #         feature = np.load(f)
        # else:
        #     print("######## Generating features from clip image encoder #########")
        #     feature = clip_imageencoder_feature(opt,self.image_paths,self.data_transforms)   
        #     print("######## Save clip image encoder features #########")

        matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.class_embedding + "_splits.mat")
        # numpy array index starts from 0, matlab starts from 1
        trainval_loc = matcontent['trainval_loc'].squeeze() - 1
        train_loc = matcontent['train_loc'].squeeze() - 1
        val_unseen_loc = matcontent['val_loc'].squeeze() - 1
        test_seen_loc = matcontent['test_seen_loc'].squeeze() - 1
        test_unseen_loc = matcontent['test_unseen_loc'].squeeze() - 1
        
        class_names = matcontent['allclasses_names']
        class_names = [name[0][0].tolist() for name in class_names]


        self.attribute = torch.from_numpy(matcontent['att'].T).float() 
        
        # shitong add here for binary attsplit
        # if opt.clip_att:
        #     matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + 'binaryAtt' + "_splits.mat")
        #     self.binary_attribute = torch.from_numpy(matcontent['att'].T).float()  # shape [50,85] in binary   


        if not opt.validation:
            if opt.preprocessing:
                if opt.standardization:
                    print('standardization...')
                    scaler = preprocessing.StandardScaler()
                else:
                    scaler = preprocessing.MinMaxScaler()
                
                # _train_feature = scaler.fit_transform(feature[trainval_loc])
                # _test_seen_feature = scaler.transform(feature[test_seen_loc])
                # _test_unseen_feature = scaler.transform(feature[test_unseen_loc])

                _train_feature = feature[trainval_loc]
                _test_seen_feature = feature[test_seen_loc]
                _test_unseen_feature = feature[test_unseen_loc]

                self.train_feature = torch.from_numpy(_train_feature).float()
                mx = self.train_feature.max()
                # self.train_feature.mul_(1/mx)   # shape 19832,2048
                self.train_feature = torch.nn.functional.normalize(self.train_feature,dim=-1)
                self.train_label = torch.from_numpy(label[trainval_loc]).long() 
                self.test_unseen_feature = torch.from_numpy(_test_unseen_feature).float()
                # self.test_unseen_feature.mul_(1/mx)
                self.test_unseen_feature = torch.nn.functional.normalize(self.test_unseen_feature,dim=-1)

                self.test_unseen_label = torch.from_numpy(label[test_unseen_loc]).long() 
                self.test_seen_feature = torch.from_numpy(_test_seen_feature).float() 
                # self.test_seen_feature.mul_(1/mx)
                self.test_seen_feature = torch.nn.functional.normalize(self.test_seen_feature,dim=-1)

                self.test_seen_label = torch.from_numpy(label[test_seen_loc]).long()

 
                self.feature = torch.from_numpy(feature).float()
                self.label = torch.from_numpy(label).long()
            else:
                self.train_feature = torch.from_numpy(feature[trainval_loc]).float()
                self.train_label = torch.from_numpy(label[trainval_loc]).long() 
                self.test_unseen_feature = torch.from_numpy(feature[test_unseen_loc]).float()
                self.test_unseen_label = torch.from_numpy(label[test_unseen_loc]).long() 
                self.test_seen_feature = torch.from_numpy(feature[test_seen_loc]).float() 
                self.test_seen_label = torch.from_numpy(label[test_seen_loc]).long()

                #shitong for test clip on the whole dataset
                self.feature = torch.from_numpy(feature).float()
                self.label = torch.from_numpy(label).long()
        else:
            self.train_feature = torch.from_numpy(feature[train_loc]).float()
            self.train_label = torch.from_numpy(label[train_loc]).long()
            self.test_unseen_feature = torch.from_numpy(feature[val_unseen_loc]).float()
            self.test_unseen_label = torch.from_numpy(label[val_unseen_loc]).long() 
    
        self.seenclasses = torch.from_numpy(np.unique(self.train_label.numpy()))
        self.unseenclasses = torch.from_numpy(np.unique(self.test_unseen_label.numpy()))
        self.ntrain = self.train_feature.size()[0]
        self.ntrain_class = self.seenclasses.size(0)
        self.ntest_class = self.unseenclasses.size(0)
        self.train_class = self.seenclasses.clone()
        self.allclasses = torch.arange(0, self.ntrain_class+self.ntest_class).long()
               
        self.train_mapped_label = map_label(self.train_label, self.seenclasses)  # change label from 绝对label 到seen class里的相对label
        
       

        # import pdb
        # pdb.set_trace()
        # print('here')
        # from torch.nn.functional import normalize
        # aa = normalize(self.feature,dim=-1)
        # from sklearn import preprocessing
        # bb = preprocessing.normalize(self.feature,norm='l2')



        # shitong
        self.all_classnames = class_names
        self.classnames = self.all_classnames
        self.unseenclassnames = [self.all_classnames[label] for label in self.unseenclasses]
        # self.allseenclasses = [self.all_classnames[label] for label in self.seenclasses]
        self.seen_unseen_classnames = [self.all_classnames[label] for label in torch.cat((self.seenclasses,self.unseenclasses),dim=0)]

        test_names = [self.all_classnames[i] for i in self.unseenclasses]
        tSNE_att_label = ['Test' if name in test_names else 'Train' for name in self.all_classnames]
        tSNE_f_label = ['Test' if label in self.unseenclasses else 'Train' for label in self.label]

        #  for tSNE visualization
        # if True:

        #     # visualize_tSNE(self.attribute,tSNE_att_label,'attribute_test')    
        #     visualize_tSNE(self.feature[:100],[self.label[:100],tSNE_f_label],['all_feature','train_test_feature'])
        #     # writer = SummaryWriter('runs/imagenet_pretrain_feature')
        #     # writer.add_embedding(self.attribute,
        #     #                     # metadata =self.all_classnames)
        #     #                     metadata = tSNE_label)
        #     # writer.close()


    def next_batch_one_class(self, batch_size):
        if self.index_in_epoch == self.ntrain_class:
            self.index_in_epoch = 0 
            perm = torch.randperm(self.ntrain_class)
            self.train_class[perm] = self.train_class[perm]

        iclass = self.train_class[self.index_in_epoch]
        idx = self.train_label.eq(iclass).nonzero().squeeze()
        perm = torch.randperm(idx.size(0))
        idx = idx[perm]
        iclass_feature = self.train_feature[idx]
        iclass_label = self.train_label[idx]
        self.index_in_epoch += 1
        return iclass_feature[0:batch_size], iclass_label[0:batch_size], self.attribute[iclass_label[0:batch_size]] 
    
    def next_batch(self, batch_size):  #used for cvpr18 xian
        idx = torch.randperm(self.ntrain)[0:batch_size]
        batch_feature = self.train_feature[idx]
        batch_label = self.train_label[idx]
        batch_att = self.attribute[batch_label]
        return batch_feature, batch_label, batch_att

    def next_batch_from_raw(self, batch_size): # shitong used for reading awa from raw
        idx = torch.randperm(self.ntrain)[0:batch_size]
        batch_path = [self.image_paths[index] for index in idx]
        batch_image = [read_image(path) for path in batch_path]
        ######## through transform and achieve batch image
        tensor_batch_image = torch.zeros((batch_size,3,224,224))
        if self.data_transforms is not None:
            for img_idx,img in enumerate(batch_image):
                tensor_batch_image[img_idx] = self.data_transforms(img)
        batch_label = self.train_label[idx]
        batch_att = self.attribute[batch_label]
        return tensor_batch_image, batch_label, batch_att

    # select batch samples by randomly drawing batch_size classes    
    def next_batch_uniform_class(self, batch_size):
        batch_class = torch.LongTensor(batch_size)
        for i in range(batch_size):
            idx = torch.randperm(self.ntrain_class)[0]
            batch_class[i] = self.train_class[idx]
            
        batch_feature = torch.FloatTensor(batch_size, self.train_feature.size(1))       
        batch_label = torch.LongTensor(batch_size)
        batch_att = torch.FloatTensor(batch_size, self.attribute.size(1))
        for i in range(batch_size):
            iclass = batch_class[i]
            idx_iclass = self.train_label.eq(iclass).nonzero().squeeze()
            idx_in_iclass = torch.randperm(idx_iclass.size(0))[0]
            idx_file = idx_iclass[idx_in_iclass]
            batch_feature[i] = self.train_feature[idx_file]
            batch_label[i] = self.train_label[idx_file]
            batch_att[i] = self.attribute[batch_label[i]] 
        return batch_feature, batch_label, batch_att

class random_loader(DATA_LOADER):
    """
    each client randomly select some images from loader, not ensure each client have all classes

    """
    def __init__(self,opt,idx,perm_list):
        super(random_loader, self).__init__(opt)
        self.ntrain = int(self.ntrain / opt.nusers)
        train_images_idx = perm_list[idx*self.ntrain:(idx+1)*self.ntrain]
        self.train_feature = self.train_feature[train_images_idx]
        self.train_label = self.train_label[train_images_idx]
        self.classnames = [self.all_classnames[label] for label in torch.unique(self.train_label)]      #shape　same as unique class label
      
class class_split_loader(DATA_LOADER):
    """
     split dataset to different clients with same number of classes
    """
    def __init__(self, opt,idx,random_class):
        super(class_split_loader,self).__init__(opt)
        class_num = int(self.seenclasses.size(0)/opt.nusers) 
        self.seenclasses=random_class[idx*class_num:(idx+1)*class_num]
        # img_idx = [np.where(self.train_label==c)[0].tolist() for c in self.seenclasses] # list two dimension not as expected
        img_idx = []
        for c in self.seenclasses:
            img_idx+=np.where(self.train_label==c)[0].tolist()
        self.ntrain = len(img_idx)
        self.train_feature = self.train_feature[img_idx]
        self.train_label = self.train_label[img_idx]
        # self.classnames = [self.all_classnames[label] for label in torch.unique(self.train_label)]
        self.classnames = [self.all_classnames[label] for label in self.seenclasses]
        self.unseenclassnames = [self.all_classnames[label] for label in self.unseenclasses]
