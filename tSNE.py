# import keyword
# import torch
# from torch.utils.tensorboard import SummaryWriter
# import torchvision.transforms as transforms
# import torchvision

# transform = transforms.Compose(
#     [transforms.ToTensor(),
#     transforms.Normalize((0.5,), (0.5,))])

# trainset = torchvision.datasets.FashionMNIST('./data',
#     download=True,
#     train=True,
#     transform=transform)
# testset = torchvision.datasets.FashionMNIST('./data',
#     download=True,
#     train=False,
#     transform=transform)

# # dataloaders
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
#                                         shuffle=True, num_workers=2)


# testloader = torch.utils.data.DataLoader(testset, batch_size=4,
#                                         shuffle=False, num_workers=2)

# # constant for classes
# classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
#         'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')

# writer = SummaryWriter('runs/fashion_mnist_experiment_1')
# def select_n_random(data, labels, n=100):
#     '''
#     Selects n random datapoints and their corresponding labels from a dataset
#     '''
#     assert len(data) == len(labels)

#     perm = torch.randperm(len(data))
#     return data[perm][:n], labels[perm][:n]

# # select random images and their target indices
# images, labels = select_n_random(trainset.data, trainset.targets)

# # get the class labels for each image
# class_labels = [classes[lab] for lab in labels]

# # log embeddings
# features = images.view(-1, 28 * 28)

# writer.add_embedding(features,
#                     metadata=class_labels,       #  text, string
#                     label_img=images.unsqueeze(1))    # image feature but shape in 28*28

# writer.close()

######################## tSNE by sklearn

from sklearn.manifold import TSNE
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
import numpy as np
import os

##　 Both X and y are array, X shape (1000,784) , y shape (1000,)
 
# X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
 
# # Randomly select 1000 samples for performance reasons
# subsample_idc = np.random.choice(X.shape[0], 1000, replace=False)
# X = X[subsample_idc,:]
# y = y[subsample_idc]

def visualize_tSNE(X,y_list,im_name_list):
# We want to get TSNE embedding with 2 dimensions
    np.random.seed(100)
    n_components = 2
    tsne = TSNE(n_components)
    tsne_result = tsne.fit_transform(X)
    # tsne_result.shape
    # (1000, 2)
    # Two dimensions for each of our images
    
    # Plot the result of our TSNE with the label color coded
    # A lot of the stuff here is about making the plot look pretty and not TSNE

    for y,im_name in zip(y_list,im_name_list):
        tsne_result_df = pd.DataFrame({'tsne_1': tsne_result[:,0], 'tsne_2': tsne_result[:,1], 'label': y}) # y also string
        plt.figure(figsize=(20,20), dpi=80)

        fig, ax = plt.subplots(1)

        sns.scatterplot(x='tsne_1', y='tsne_2', hue='label', data=tsne_result_df, ax=ax,s=20)
        lim = (tsne_result.min()-5, tsne_result.max()+5)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_aspect('equal')
        ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
        im_name = os.path.join('./tSNE_images',im_name)
        print('im_path',im_name)
        plt.savefig(im_name)




def visualize_tSNE_2label(X,y_list,im_name_list,class_label_list):
# We want to get TSNE embedding with 2 dimensions
    np.random.seed(100)
    n_components = 2
    tsne = TSNE(n_components)
    tsne_result = tsne.fit_transform(X)
    # tsne_result.shape
    # (1000, 2)
    # Two dimensions for each of our images
    
    # Plot the result of our TSNE with the label color coded
    # A lot of the stuff here is about making the plot look pretty and not TSNE

    for y,im_name,class_label in zip(y_list,im_name_list,class_label_list):
        tsne_result_df = pd.DataFrame({'tsne_1': tsne_result[:,0], 'tsne_2': tsne_result[:,1], 'label': y, 'label_class': class_label}) # y also string
        plt.figure(figsize=(20,20), dpi=80)

        fig, ax = plt.subplots(1)

        sns.scatterplot(x='tsne_1', y='tsne_2', hue='label_class',style='label', data=tsne_result_df, ax=ax,s=20)
        lim = (tsne_result.min()-5, tsne_result.max()+5)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_aspect('equal')
        ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
        im_name = os.path.join('./tSNE_images',im_name)
        print('im_path',im_name)
        plt.savefig(im_name)