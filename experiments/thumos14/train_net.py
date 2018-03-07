#!/usr/bin/env python

# --------------------------------------------------------
# R-C3D
# Copyright (c) 2017 Boston Univ.
# Licensed under The MIT License [see LICENSE for details]
# Written by Huijuan Xu
# --------------------------------------------------------


import _init_paths
import caffe
import argparse
import pprint
import numpy as np
import sys
import cPickle
import copy

from tdcnn.config import cfg, cfg_from_file, cfg_from_list
from tdcnn.train import get_training_roidb, train_net

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a R-C3D network')
    parser.add_argument('--gpu', dest='gpu_id',
                        help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--solver', dest='solver',
                        help='solver prototxt',
                        default=None, type=str)
    parser.add_argument('--iters', dest='max_iters',
                        help='number of iterations to train',
                        default=60000, type=int)
    parser.add_argument('--weights', dest='pretrained_model',
                        help='initialize with pretrained model weights',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default=None, type=str)
    parser.add_argument('--rand', dest='randomize',
                        help='randomize (do not use a fixed seed)',
                        action='store_true')
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


def get_val_roidb(path):
    val_data = cPickle.load(open(path + 'val/val_data_512.pkl'))
    return val_data

if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    cfg.GPU_ID = args.gpu_id

    print('Using config:')
    pprint.pprint(cfg)

    if not args.randomize:
        # fix the random seeds (numpy and caffe) for reproducibility
        np.random.seed(cfg.RNG_SEED)
        caffe.set_random_seed(cfg.RNG_SEED)

    # set up caffe
    caffe.set_device(args.gpu_id)
    caffe.set_mode_gpu()

    path = './preprocess/'
    roidb = get_val_roidb(path)
    print '{:d} roidb entries'.format(len(roidb))


    output_dir = './experiments/thumos14/snapshot/'
    print 'Output will be saved to `{:s}`'.format(output_dir)

    train_net(args.solver, roidb, output_dir,
              pretrained_model=args.pretrained_model,
              max_iters=args.max_iters)

