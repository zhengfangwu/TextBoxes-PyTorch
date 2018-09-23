import torch
import torch.utils.data

import os
from skimage import io, transform

vgg_mean = torch.Tensor([0.485, 0.456, 0.406])
vgg_std = torch.Tensor([0.229, 0.224, 0.225])

class ICDARLabel(object):

    def __init__(self, idx, string):
        super(ICDARLabel, self).__init__()
        elems = string.strip().split()
        self.idx = idx
        self.bbox = []
        for i in range(4):
            self.bbox.append(int(elems[i]))
        self.bbox = torch.Tensor(self.bbox)
        self.text = elems[4]
    
    def __str__(self):
        return 'Img: %s Label: %s at (%d, %d, %d, %d)' % (self.idx, self.text,\
            self.bbox[0].item(), self.bbox[1].item(), self.bbox[2].item(), self.bbox[3].item())

class ICDARDataset(torch.utils.data.Dataset):

    def __init__(self, img_path, gt_path, img_h, img_w, use_cuda=True, mean=vgg_mean, std=vgg_std):
        super(ICDARDataset, self).__init__()

        self.img_path = img_path
        self.gt_path = gt_path
        self.use_cuda = use_cuda
        self.mean = mean.cuda() if use_cuda else mean
        self.std = std.cuda() if use_cuda else std

        # prepare training datasets
        img_list = os.listdir(img_path)
        self.img = []
        self.gt = []
        for img_file in img_list:

            img = io.imread(os.path.join(img_path, img_file))
            img = transform.resize(img, (img_h, img_w))
            img_t = torch.Tensor(img).float()
            self.img.append(img_t)
            
            img_idx, _ = os.path.split(img_file)
            gt_file = os.path.join(gt_path, 'gt_'+img_idx+'.txt')
            with open(gt_file) as f:
                gt_int = f.readlines()
            gt_int = [ICDARLabel(img_idx, x) for x in gt_int]
            self.gt.append(gt_int)

    
    def __len__(self):
        return len(self.img)

    def image_augmentation(img):
        

    def __getitem__(self, idx):
        img_t = self.img[idx]
        bbox_t = self.gt[idx].bbox
        if self.use_cuda:
            img_t = img_t.cuda()
            bbox_t = bbox_t.cuda()
        img_t = (img_t / 255.0 - self.mean) / self.std
        return img_t, bbox_t