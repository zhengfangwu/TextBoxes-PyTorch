import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim

from src import *

parser = argparse.ArgumentParser(description='Training argument.')
parser.add_argument('--train_img_path', type=str, required=True)
parser.add_argument('--train_gt_path', type=str, required=True)
parser.add_argument('--test_img_path', type=str, required=True)
parser.add_argument('--test_gt_path', type=str, required=True)
parser.add_argument('--train_batch_size', type=int, required=True)
parser.add_argument('--test_batch_size', type=int, default=1)
parser.add_argument('--shuffle', type=int, default=True)
parser.add_argument('--use_cuda', type=int, default=True)
parser.add_argument('--data_threads', type=int, default=1)
parser.add_argument('--epoches', type=int, required=True)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--cuda', type=int, default=True)
args = parser.parse_args()
args.shuffle = True if args.shuffle == 1 else False
args.use_cuda = True if args.use_cuda == 1 else False
print(args)

img_h = 300
img_w = 300
# 20% ~ 95% of image size
min_size = [30, 60, 114, 168, 222, 276] 
max_size = [None, 114, 167, 222, 276, 330]
aspect_ratios = [2, 3, 5, 7, 10]
threshold = 0.5
variances = [0.1, 0.1, 0.2, 0.2]
neg_ratio = 3.0

train_dataset = ICDARDataset(args.train_img_path, args.train_gt_path, img_h, img_w, use_cuda=args.use_cuda)
test_dataset = ICDARDataset(args.test_img_path, args.test_gt_path, img_h, img_w, use_cuda=args.use_cuda)
train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=args.shuffle, num_workers=args.data_threads)
test_dataloader = DataLoader(test_dataset, batch_size =args.test_batch_size, shuffle=False)

net = Net(min_size, max_size, aspect_ratios, use_cuda=args.cuda)
net.apply(init_weights)
criterion = MultiBoxLoss(threshold, variances, neg_ratio, use_cuda=args.use_cuda)
optimizer = optim.Adam(net.parameters(), lr=args.lr)

if args.cuda:
    net.cuda()
    criterion.cuda()

def train(epoch_idx):
    net.train()
    loss_loc = 0.0
    loss_conf = 0.0

    for idx, (images, targets) in enumerate(train_dataloader):

        optimizer.zero_grad()
        
        if args.cuda:
            images = images.cuda()
            targets = targets.cuda()
        
        pred = net(images)

        loss_l, loss_c = criterion(pred, targets)
        loss = loss_l + loss_c
        loss.backward()
        optimizer.step()

        print('Epoch %d[%d/%d] loss_l: %.6f loss_c: %.6f' \
            % (epoch_idx, idx, len(train_dataloader), \
            loss_l.detach().item(), loss_c.detach().item()))

def test(epoch_idx):
    net.eval()

if __name__ == "__main__":
    for i in range(1, args.epoches+1):
        train(i)
        test(i)

