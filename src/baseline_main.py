#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

from options import args_parser
from models import get_net

args = args_parser()
args.model='cnn'

if __name__ == '__main__':
    #args = args_parser()
    if args.model == 'cnn':
        a=get_net('LeNet5')
