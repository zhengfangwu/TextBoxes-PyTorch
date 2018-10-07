import argparse
import pprint

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim

from src import *

parser = argparse.ArgumentParser(description='Training argument.')

# dataset configuration
parser.add_argument('--train_img_path', type=str, required=True)
parser.add_argument('--train_gt_path', type=str, required=True)
parser.add_argument('--test_img_path', type=str, required=True)
parser.add_argument('--test_gt_path', type=str, required=True)
parser.add_argument('--shuffle', type=int, default=True)
parser.add_argument('--data_threads', type=int, default=0)
# dataset image
parser.add_argument('--img_h', type=int, default=300)
parser.add_argument('--img_w', type=int, default=300)

# model init
parser.add_argument('--load_vgg', type=int, default=1)
parser.add_argument('--resume', type=int, default=0)
parser.add_argument('--resume_checkpoint_folder', type=str)

# training
parser.add_argument('--train_batch_size', type=int, required=True)
parser.add_argument('--test_batch_size', type=int, default=1)
parser.add_argument('--use_cuda', type=int, default=True)
parser.add_argument('--epoches', type=int, required=True)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--cuda', type=int, default=True)
parser.add_argument('--cuda_device_id', type=int, default=-1)
parser.add_argument('--display_interval', type=int, default=10)
# checkpoint
parser.add_argument('--save_checkpoint_interval', type=int, default=1)
parser.add_argument('--save_checkpoint_folder', type=str, default='./checkpoint')


args = parser.parse_args()
args.shuffle = True if args.shuffle == 1 else False
if args.cuda == 1:
    if args.cuda_device_id == -1:
        device = torch.device('cuda')
    else:
        device = torch.device('cuda:' + str(args.cuda_device_id))
else:
    device = torch.device('cpu')
pprint.pprint(args)

# 20% ~ 95% of image size
min_size = [30, 60, 114, 168, 222, 276] 
max_size = [None, 114, 167, 222, 276, 330]
aspect_ratios = [2, 3, 5, 7, 10]
threshold = 0.5
variances = torch.tensor([0.1, 0.1, 0.2, 0.2]).to(device)

if not os.path.exists(args.save_checkpoint_folder):
    os.mkdir(args.save_checkpoint_folder)
neg_ratio = 3.0

train_dataset = ICDARDataset(args.train_img_path, args.train_gt_path, args.img_h, args.img_w, use_cuda=args.use_cuda)
test_dataset = ICDARDataset(args.test_img_path, args.test_gt_path, args.img_h, args.img_w, use_cuda=args.use_cuda)
train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=args.shuffle, num_workers=args.data_threads)
test_dataloader = DataLoader(test_dataset, batch_size =args.test_batch_size, shuffle=False)

net = Net(min_size, max_size, aspect_ratios, use_cuda=args.cuda)
criterion = MultiBoxLoss(threshold, variances, neg_ratio, use_cuda=args.use_cuda)
optimizer = optim.Adam(net.parameters(), lr=args.lr)

if args.resume == 0:
    initialize(net)
else:
    args.epoches = load_checkpoint(args.resume_checkpoint_folder, net, optimizer)

net.to(device)
criterion.to(device)
optimizer.to(device)

def train(epoch_idx):
    net.train()
    loss_loc = 0.0
    loss_conf = 0.0

    for idx, (images, targets) in enumerate(train_dataloader, 1):

        optimizer.zero_grad()
        
        images = images.to(device)
        targets = targets.to(device)
        
        pred = net(images)

        loss_l, loss_c = criterion(pred, targets)
        loss = loss_l + loss_c
        loss.backward()
        optimizer.step()

        loss_loc += loss_l.detach().item()
        loss_conf += loss_c.detach().item()

        if idx % args.display_interval == 0:
            print('Epoch %d[%d/%d] loss_loc: %.6f loss_conf: %.6f' \
                % (epoch_idx, idx, len(train_dataloader), \
                loss_loc / args.display_interval, loss_conf / args.display_interval))

def test(epoch_idx):
    net.eval()
    loss_loc = 0.0
    loss_conf = 0.0
    loss_total = 0.0

    with torch.no_grad():
        for idx, (images, targets) in enumerate(test_dataloader, 1):

            images = images.to(device)
            targets = targets.to(device)

            pred = net(images)

            loss_l, loss_c = criterion(pred, targets)
            loss = loss_l + loss_c

            loss_loc += loss_l.detach().item()
            loss_conf += loss_c.detach().item()
            loss_total += loss.detach().item()
        
        print('Test Epoch %d: loss_loc: %.6f loss_conf: %.6f loss_total %.6f' \
            % (epoch_idx, loss_loc / len(test_dataloader),
                loss_conf / len(test_dataloader),
                loss_total / len(test_dataloader)))

        


if __name__ == "__main__":
    for i in range(1, args.epoches+1):
        train(i)
        test(i)
        save_checkpoint(i, args.save_checkpoint_folder, net, optimizer)

