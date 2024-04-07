import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.metrics import confusion_matrix

def viusal_confusion_matrix(y_pred,y_real,categories,im_name,showvalue=True):
    
    cfs_mtrix = confusion_matrix(y_real, y_pred)
    # want to show in percent
    cfs_mtrix = cfs_mtrix/np.sum(cfs_mtrix,axis=1).reshape(-1,1)

    fig, ax = plt.subplots(1)
    plt.figure(figsize=(35,25))

    sns.set(font_scale=2.4)
    ax.set_yticklabels(categories, rotation=45)
    ax.set_xticklabels(categories, rotation=45)
    if showvalue:
        p = sns.heatmap(cfs_mtrix, fmt='.2%',annot=True,xticklabels=categories,yticklabels=categories, cmap = "Blues",square=True,vmin=0)
    else:
        p = sns.heatmap(cfs_mtrix, fmt='.2%',xticklabels=categories,yticklabels=categories, cmap = "Blues",square=True,vmin=0)

    p.set_xlabel("Predicted label", fontsize = 40)
    p.set_ylabel("True label", fontsize = 40)

    # plt.ylabel('True label')
    # plt.xlabel('Predicted label')
    ax.set_aspect('equal')


    im_name = os.path.join('./cfs_images',im_name)
    print('confusion_matrix path',im_name)
    plt.savefig(im_name)

# if __name__ =='__main__':
#     y_pred = np.array([1,2,1,2,1,2,3,4,1,9])
#     y_real = np.array([1,2,4,2,7,1,3,4,5,9])
#     categories = ["aa","s","c","ww","vv","b","qw","v","q","b"]
#     viusal_confusion_matrix(y_pred,y_real,categories,'debug_2')

    # sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, 
    #             fmt='.2%', cmap='Blues')

# group_names = ['True Neg','False Pos','False Neg','True Pos']
# group_counts = ["{0:0.0f}".format(value) for value in
#                 cf_matrix.flatten()]
# group_percentages = ["{0:.2%}".format(value) for value in
#                      cf_matrix.flatten()/np.sum(cf_matrix)]
# labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names,group_counts,group_percentages)]
# labels = np.asarray(labels).reshape(2,2)


# lim = (cf_matrix.min()-5, cf_matrix.max()+5)
# ax.set_xlim(lim)
# ax.set_ylim(lim)
