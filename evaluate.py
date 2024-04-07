#
# Evaluating with saved features
#

import scipy.io
import torch
import numpy as np
import os
from evaluation.compute_acc import compute_acc,compute_acc_visualize

import matplotlib.pyplot as plt
from PIL import Image,ImageDraw


# set gpu device
def evaluate():
    torch.cuda.set_device(0)

    # load features and labels
    result = scipy.io.loadmat('result_feature.mat')
    query_feature = torch.FloatTensor(result['query_f'])
    query_cam = result['query_cam'][0]
    query_label = result['query_label'][0]
    gallery_feature = torch.FloatTensor(result['gallery_f'])
    gallery_cam = result['gallery_cam'][0]
    gallery_label = result['gallery_label'][0]

    query_feature = query_feature.cuda()
    gallery_feature = gallery_feature.cuda()

    # compute rank accuracy and mAP
    CMC = torch.IntTensor(len(gallery_label)).zero_()
    ap = 0.0
    for i in range(len(query_label)):
    # print('Process count:', i)

        ap_tmp, CMC_tmp = compute_acc(query_feature[i],query_label[i],query_cam[i],
                                        gallery_feature,gallery_label,gallery_cam)
        if CMC_tmp[0]==-1:
            continue
        CMC = CMC + CMC_tmp
        ap += ap_tmp


    CMC = CMC.float()
    CMC = CMC/len(query_label) #average CMC
    print('Rank@1:%f Rank@5:%f Rank@10:%f mAP:%f'%(CMC[0],CMC[4],CMC[9], ap/len(query_label)))
    
    return CMC[0],ap/len(query_label)

def frame_image(img, frame_width):
    img = np.array(img)
    b = frame_width # border size in pixel
    ny, nx = img.shape[0], img.shape[1] # resolution / number of pixels in x and y
    if img.ndim == 3: # rgb or rgba array
        framed_img = np.zeros((b+ny+b, b+nx+b, img.shape[2]))
    elif img.ndim == 2: # grayscale image
        framed_img = np.zeros((b+ny+b, b+nx+b))
    framed_img[b:-b, b:-b] = img
    return framed_img

def visualize_ranking_list(query_path,retrivial_path,query_label,retrivial_labels,key):
    fig = plt.figure(figsize=(16,4))
    ax = plt.subplot(1,13,1)
    ax.axis('off')
    # ax.tick_params(color='green', labelcolor='green')

    query_path = query_path.strip()
    q_img = Image.open(query_path)
    plt.imshow(q_img)

    for i in range(12): #Show top-10 images
        ax = plt.subplot(1,13,i+2)
        ax.axis('off')
        img_path = retrivial_path[i][0].strip()
        g_img = Image.open(img_path)
        plt.imshow(g_img,zorder=10)

        g_label = retrivial_labels[i]
        if g_label == query_label:
            # green = np.zeros((g_img.size[1]+5,g_img.size[0]+5))
            # green = Image.fromarray(green)
            # plt.imshow(green,zorder = 0)
            ax.set_title('%d'%(i+1), color='black') # true matching
        else:
            ax.set_title('%d'%(i+1), color='black') # false matching
        print(img_path)
        # plt.imshow(g_img)
    img_name = './'+key+'.png'
    print(img_name) 
    plt.savefig(img_name)
    
    return

def evaluate_visualize(args):   ####change to one image query
    SKA_flag = args.SKA
    i = args.image_id
    print(i)
    torch.cuda.set_device(0)

    # load features and labels
    
    result_dict = {'fedavg':scipy.io.loadmat('result_feature_baseline.mat'),
    'fedreid':scipy.io.loadmat('result_feature_fedreid_market.mat'),
    'SKA':scipy.io.loadmat('result_feature_SKA.mat')}

    # result_dict = {'fedavg':scipy.io.loadmat('result_feature_baseline.mat')}

    for (key,result) in result_dict.items():

        query_feature = torch.FloatTensor(result['query_f'])
        query_cam = result['query_cam'][0]
        query_label = result['query_label'][0]
        gallery_feature = torch.FloatTensor(result['gallery_f'])
        gallery_cam = result['gallery_cam'][0]
        gallery_label = result['gallery_label'][0]
        gallery_path = result['gallery_path']
        query_path = result['query_path']       
        query_feature = query_feature.cuda()
        gallery_feature = gallery_feature.cuda()

        # compute rank accuracy and mAP
        CMC = torch.IntTensor(len(gallery_label)).zero_()
        ap = 0.0
        # print('Process count:', i)

        ap_tmp, CMC_tmp, index = compute_acc_visualize(query_feature[i],query_label[i],query_cam[i],
                                        gallery_feature,gallery_label,gallery_cam)
        if CMC_tmp[0]==-1:
            print('no corresponding correct image in gallery')
        #     continue
        # CMC = CMC + CMC_tmp
        # ap += ap_tmp

        # CMC = CMC.float()
        # CMC = CMC/len(query_label) #average CMC
        # print('Rank@1:%f Rank@5:%f Rank@10:%f mAP:%f'%(CMC[0],CMC[4],CMC[9], ap/len(query_label)))
        retrivial_images_path = gallery_path[index[:15]][:,0]
        query_image_path = query_path[i][0]
        visualize_ranking_list(query_image_path,retrivial_images_path,query_label[i], gallery_label[index],key)

    return 

# def test():
   
#     w, h = 220, 190
#     shape = [(40, 40), (w - 10, h - 10)]
    
#     # creating new Image object
#     img = Image.new("RGB", (w, h))
    
#     # create rectangle image
#     img1 = ImageDraw.Draw(img)  
#     img1.rectangle(shape, fill ="#ffff33", outline ="red")
#     import pdb
#     pdb.set_trace()
#     plt.imshow(img1)
#     plt.savefig('./foo.png')
#     return



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--SKA', default=1, type=int,help='GPU IDs: e.g. 0, 1, 2, ...')
    parser.add_argument('--image_id', default=100, type=int,help='GPU IDs: e.g. 0, 1, 2, ...')

    args = parser.parse_args()

    evaluate_visualize(args)
