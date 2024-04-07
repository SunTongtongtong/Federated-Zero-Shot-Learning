# # encoding: utf-8
# """
# @author:  liaoxingyu
# @contact: xyliao1993@qq.com 
# """

# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function
# from __future__ import unicode_literals

# import math

# import numpy as np


# class AverageMeter(object):
#     def __init__(self, name):
#         self.name = name
#         self.n = 0
#         self.sum = 0.0
#         self.var = 0.0
#         self.val = 0.0
#         self.mean = np.nan
#         self.std = np.nan

#     def update(self, value, n=1):
#         self.val = value
#         self.sum += value
#         self.var += value * value
#         self.n += n

#         if self.n == 0:
#             self.mean, self.std = np.nan, np.nan
#         elif self.n == 1:
#             self.mean, self.std = self.sum, np.inf
#         else:
#             self.mean = self.sum / self.n
#             self.std = math.sqrt(
#                 (self.var - self.n * self.mean * self.mean) / (self.n - 1.0))

#     def value(self):
#         return self.mean, self.std

#     def get(self):
#         return self.name, self.mean

#     def reset(self):
#         self.n = 0
#         self.sum = 0.0
#         self.var = 0.0
#         self.val = 0.0
#         self.mean = np.nan
#         self.std = np.nan
from __future__ import absolute_import
from __future__ import division


class AverageMeter(object):
    """Computes and stores the average and current value.
       
       Code imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self,name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def __str__(self):
        fmtstr = '{name} {val' + self.fmt +'} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)