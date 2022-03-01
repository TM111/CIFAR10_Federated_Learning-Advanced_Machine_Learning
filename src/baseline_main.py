#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
from sampling import dirichlet_distribution, cifar_iid, cifar_noniid
from utils import get_dataset
from options import args_parser
from models import get_net

args = args_parser()
if(args.colab==False):
    args.model='cnn'

if __name__ == '__main__':
    if(args.colab==True):
        args = args_parser()
    if args.model == 'cnn':
        a=get_net('LeNet5')
        print(a)