import torch
import torch.utils.data

import os
import random
import numpy as np
from skimage import io, transform, color

vgg_mean = torch.Tensor([0.485, 0.456, 0.406])
vgg_std = torch.Tensor([0.229, 0.224, 0.225])

def jaccard_numpy(box_a, box_b):
    max_xy = np.minimum(box_a[:, 2:], box_b[2:])
    min_xy = np.maximum(box_a[:, :2], box_b[:2])
    inter = np.clip((max_xy - min_xy), a_min = 0, a_max = np.inf)
    inter = inter[:, 0] * inter[:, 1]

    area_a = ((box_a[:, 2] - box_a[:, 0]) *
              (box_a[:, 3] - box_a[:, 1]))
    area_b = ((box_b[2] - box_b[0]) *
              (box_b[3] - box_b[1]))
    union = area_a + area_b - inter
    return inter / union

class ICDARLabel(object):

    def __init__(self, idx, string):
        super(ICDARLabel, self).__init__()
        elems = string.strip().split()
        self.idx = idx
        self.bbox = []
        for i in range(4):
            self.bbox.append(int(elems[i]))
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
        self.img_list = os.listdir(img_path)

    def __len__(self):
        return len(self.img_list)

    def image_augmentation(self, image, boxes):
        # ConvertFromInts
        image = image.astype(np.float32)

        # ToAbsoluteCoords (ignored)

        # ------------ PhotometricDistort --------------
        # RandomBrightness
        delta = 32
        if random.randint(0, 1):
            image += random.uniform(-delta, delta)

        r = random.randint(0, 1)
        if r == 1:
            # RandomContrast first
            lower = 0.5
            upper = 1.5
            if random.randint(0, 1):
                image *= random.uniform(lower, upper)

        # ConvertColor: convert from RGB to HSV
        color.rgb2hsv(image)

        # RandomSaturation
        lower = 0.5
        upper = 1.5
        if random.randint(0, 1):
            image[:, :, 1] *= random.uniform(lower, upper)

        # RandomHue
        delta = 18.0
        if random.randint(0, 1):
            image[:, :, 0] += random.uniform(-delta, delta)
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
        
        # ConvertColor: convert from HSV to RGB
        color.hsv2rgb(image)

        if r == 0:
            # RandomContrast later
            lower = 0.5
            upper = 1.5
            if random.randint(0, 1):
                image *= random.uniform(lower, upper)
        
        # RandomLightingNoise
        perms = ((0, 1, 2), (0, 2, 1),
                 (1, 0, 2), (1, 2, 0),
                 (2, 0, 1), (2, 1, 0))

        if random.randint(0, 1):
            swap = perms[random.randint(0, len(perms) - 1)]
            image = image[:, :, swap]

        # Expand(mean)
        mean = (104, 117, 123)
        if random.randint(0, 1):
            
            height, width, depth = image.shape
            ratio = random.uniform(1, 4)
            left = random.uniform(0, width * ratio - width)
            top = random.uniform(0, height * ratio - height)

            expand_image = np.zeros(
                (int(height * ratio), int(width * ratio), depth),
                dtype = image.dtype
            )
            expand_image[:, :, :] = mean
            expand_image[int(top):int(top + height),
                   int(left):int(left + width)] = image
            image = expand_image

            boxes[:, :2] += (int(left), int(top))
            boxes[:, 2:] += (int(left), int(top))

        # RandomSampleCrop
        sample_options = (
            None,           # use entirely original image
            (0.1, None),    # sample a patch with jaccard w/obj in 0.1, 0.3, 0.5, 0.7, 0.9
            (0.3, None),    
            (0.7, None),
            (0.9, None),
            (None, None)    # randomly sample a patch
        )

        height, width, _ = image.shape
        while True:
            flag = True
            mode = random.choice(sample_options)
            if mode is None:
                break
            
            min_iou, max_iou = mode
            if min_iou is None:
                min_iou = float('-inf')
            if max_iou is None:
                max_iou = float('inf')
            
            for _ in range(50):
                current_image = image

                w = random.uniform(0.3 * width, width)
                h = random.uniform(0.3 * height, height)

                # aspect ratio constraint between 0.5 ~ 2
                if h / w < 0.5 or h / w > 2:
                    continue
                
                left = random.uniform(0, width - w)
                top = random.uniform(0, height - h)

                rect = np.array([int(left), int(top), int(left + w), int(top + h)])
                # Note: actually, "top" is bottom axis
                overlap = jaccard_numpy(boxes, rect)

                if overlap.min() < min_iou and max_iou < overlap.max():
                    continue
                
                current_image = current_image[rect[1]:rect[3],
                                              rect[0]:rect[2],
                                              :]
                
                centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0

                # mask in all gt boxes whose centers are over the sampled rectangle
                mask = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1]) \
                   * (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])
                
                if not mask.any():
                    continue
                
                current_boxes = boxes[mask, :].copy()
                current_boxes[:, :2] = np.maximum(current_boxes[:, :2], rect[:2])
                current_boxes[:, :2] -= rect[:2]
                current_boxes[:, 2:] = np.minimum(current_boxes[:, 2:], rect[2:])
                current_boxes[:, 2:] -= rect[:2]

                image = current_image
                boxes = current_boxes
                flag = False
                break
            
            if flag is False:
                break

        # RandomMirror
        if random.randint(0, 1):
            image = image[:, ::-1]
            boxes = boxes.copy()
            boxes[:, 0::2] = width - boxes[:, 2::-2]

        # ToPercentCoords (ignored)

        # Resize(size)
        size = 300
        image = transform.resize(image, (size, size))

        # SubtractMeans(mean)
        image = (image / 255.0 - vgg_mean) / vgg_std

        return image, boxes

    def __getitem__(self, idx):
        
        img_file = self.img_list[idx]
        image = io.imread(os.path.join(self.img_path, img_file))
        
        img_idx, _ = os.path.split(img_file)
        gt_file = os.path.join(self.gt_path, 'gt_'+img_idx+'.txt')
        with open(gt_file) as f:
            gt_int = f.readlines()
        gt = [ICDARLabel(img_idx, x) for x in gt_int]
        boxes = np.array([x.bbox for x in gt])
        image, boxes = self.image_augmentation(image, boxes)
        image_t = torch.Tensor(image)
        boxes_t = torch.Tensor(boxes)
        if self.use_cuda:
            image_t = image_t.cuda()
            boxes_t = boxes_t.cuda()
        return image_t, boxes_t