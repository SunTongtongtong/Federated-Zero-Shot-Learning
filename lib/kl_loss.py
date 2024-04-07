#
# KL loss for knowledge distillation
#

from __future__ import absolute_import

import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

class KLLoss(nn.Module):
    def __init__(self):

        super(KLLoss, self).__init__()
    def forward(self, pred, label, T):
        # pred: 2D matrix (batch_size, num_classes)
        # label: 1D vector indicating class number
        predict = F.log_softmax(pred/T,dim=1)
        target_data = F.softmax(label/T,dim=1)
        target_data =target_data+10**(-7)
        target = Variable(target_data.data.cuda(),requires_grad=False)
        loss=T*T*((target*(target.log()-predict)).sum(1).sum()/target.size()[0])

        return loss

