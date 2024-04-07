#
# Model testing parameter setting
#

import argparse
import os
######################################################################
# Options
# --------
# Hyper-parameters (consistent with training setting)
parser = argparse.ArgumentParser(description='Testing')
parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--test_data_dir',type=str, help='path of testing dataset',
    default=os.path.join(os.path.dirname(os.path.abspath(__file__)),'targetDataset/QMUL-iLIDS/pytorch'))
parser.add_argument('--batchsize', default=32, type=int, help='batchsize')

parser.add_argument('--frac', type=str, default='epoch100', help='the fraction of clients: C')

parser.add_argument('--agg', type=str, default='avg', help='Federated average strategy')
parser.add_argument('--name',default='withoutExpert_2nd', type=str, help='Model Name')
parser.add_argument('--logs_dir', type=str, help='path of logs',
    default=os.path.join(os.path.dirname(os.path.abspath(__file__)),'model_save'))

parser.add_argument('--model_name',default='model_SNR.03_13_21:59:10.pth', type=str, help='Model Name')

parser.add_argument('--AN', action='store_false', help='use all training data' )
parser.add_argument("--attr_num", type=int, default=39)


opt = parser.parse_args()


def image_dataset_kwargs(parsed_args):
    """
    Build kwargs for ImageDataManager in data_manager.py from
    the parsed command-line arguments.
    """
    return {
        'source_names': parsed_args.source_names,
        'target_names': parsed_args.target_names,
        'finetune_names': parsed_args.finetune_names,
        'root': parsed_args.root,
        'split_id': parsed_args.split_id,
        'height': parsed_args.height,
        'width': parsed_args.width,
        'train_batch_size': parsed_args.train_batch_size,
        'test_batch_size': parsed_args.test_batch_size,
        'workers': parsed_args.workers,
        'train_sampler': parsed_args.train_sampler,
        'num_instances': parsed_args.num_instances,

    }