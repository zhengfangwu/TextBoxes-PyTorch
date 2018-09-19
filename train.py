import argparse

import torch
from torch.utils.data import DataLoader

from src import *

parser = argparse.ArgumentParser(description='Training argument.')
parser.add_argument('train_img_path', type=str, required=True)
parser.add_argument('train_gt_path', type=str, required=True)
parser.add_argument('test_img_path', type=str, required=True)
parser.add_argument('test_gt_path', type=str, required=True)
parser.add_argument('batch_size', type=int, required=True)
parser.add_argument('shuffle', type=int, default=True)
parser.add_argument('use_cuda', type=int, default=True)
parser.add_argument('data_threads', type=int, default=4)
parser.add_argument('epoches', type=int, required=True)
args = parser.parse_args()
args.shuffle = True if args.shuffle == 1 else False
args.use_cuda = True if args.use_cuda == 1 else False

img_h = 300
img_w = 300
# 20% ~ 95% of image size
min_size = [30, 60, 114, 168, 222, 276] 
max_size = [None, 114, 167, 222, 276, 330]
aspect_ratios = [2, 3, 5, 7, 10]

train_dataset = ICDARDataset(args.train_img_path, args.train_gt_path, img_h, img_w, use_cuda=args.use_cuda)
test_dataset = ICDARDataset(args.test_img_path, args.test_gt_path, img_h, img_w, use_cuda=args.use_cuda)
train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=args.data_threads)
test_dataloader = DataLoader(test_dataset, batch_size = 1, shuffle=False)



net = Net(min_size, max_size, aspect_ratios)
