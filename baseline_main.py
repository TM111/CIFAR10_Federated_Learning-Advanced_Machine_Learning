#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


from options import args_parser

if __name__ == '__main__':
    args = args_parser()
    if args.model == 'cnn':
      print(23)