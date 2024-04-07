from turtle import pd
from torchvision import datasets
import os
import numpy as np
import pickle
import pdb
class attribute_dataset(datasets.ImageFolder):

    def __init__(self,root, loader):
        super(attribute_dataset,self).__init__(root, loader)
        dataset_name = root.split('/')[-3]+'_one_hot_smaller_attr_label.pkl'
        label_dir_base = '/import/sgg-homes/ss014/project/zerodim'
        label_path = os.path.join(label_dir_base,dataset_name)
        # self.label = np.load(label_path,allow_pickle=True)
        with open(label_path,'rb') as f:
            self.label = pickle.load(f)
        
        #
        index = 5
        self.attr_text(index)
        print('gg')


    def __getitem__(self, index) :
        path, _ = self.samples[index]
        attr_label = self.label[path]

        return super().__getitem__(index),attr_label


    def attr_text(self,index):
        path, _ = self.samples[index]
        attr_label = self.label[path]
        attr_label = attr_label.astype(np.bool)

        values_names = ["a kid","a teenager","an adult","an old person",
                "a male","a female",
                "short hair","long hair",
                "long sleeve","short sleeve",
                "long lower body clothing","short lower body clothing",
                "dress","pants",
                "wear hat","no hat",
                "wear backpack","no backpack",
                "wear bag","no bag",
                "wear handbag","no handbag",
                "up black","up white","up red","up purple","up yellow","up gray","up blue","up green",
                "down black","down white","down pink","down purple","down yellow","down gray","down blue","down green","down brown",
                ]
        print(path)
        values_names = np.array(values_names)
        print(values_names[attr_label])