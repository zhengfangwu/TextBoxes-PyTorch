import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class L2Norm(nn.Module):

    def __init__(self, in_channels, scale_init):
        super(L2Norm, self).__init__()
        self.in_channels = in_channels
        self.scale_init = scale_init
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.in_channels))       
        self.init_weight()
 
    def init_weight(self):
        torch.nn.init.constant_(self.weight, self.scale_init)
    
    def forward(self, input):
        norm = input.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps
        norm_input = torch.div(input, norm)
        out = self.weight.view(1, -1, 1, 1).expand_as(input) * norm_input
        return out
    

class PriorBoxLayer(nn.Module):
    
    # note: flip not implemented
    # variance is not needed to generate prior boxes
    def __init__(self, min_size, max_size, aspect_ratios, clip, use_cuda):
        super(PriorBoxLayer, self).__init__()

        self.min_size = min_size
        self.max_size = max_size
        self.aspect_ratios = aspect_ratios
        self.clip = clip
        self.use_cuda = use_cuda

        self.num_priors = len(aspect_ratios) + 1
        if max_size is not None:
            self.num_priors += 1

        self.priors = None

    def forward(self, img, feature):
        img_height = img.size(2)
        img_width = img.size(3)
        layer_height = feature.size(2)
        layer_width = feature.size(3)
        step_x = 1.0 * img_width / layer_width
        step_y = 1.0 * img_height / layer_height

        # [ (1), (2), layer_h, layer_w, (num_priors), (4) ]
        # (center_coord + add_mask * box_dim / 2.) / img_dim
        # max parallalism
        if self.priors == None:

            # center_coord
            w = torch.arange(0, layer_width).view(1, -1).expand(layer_height, layer_width).float()
            h = torch.arange(0, layer_height).view(-1, 1).expand(layer_height, layer_width).float()
            if self.use_cuda:
                w = w.cuda()
                h = h.cuda()
            center_x = (w + 0.5) * step_x
            center_y = (h + 0.5) * step_y
            center_coord = torch.stack((center_x, center_y, center_x, center_y), dim=2)
            center_coord = center_coord.unsqueeze(2).expand(-1, -1, self.num_priors, -1)

            # add_mask
            mask_add = torch.tensor(1.0).view(1, 1, 1)
            mask_minus = torch.tensor(-1.0).view(1, 1, 1)
            if self.use_cuda:
                mask_add = mask_add.cuda()
                mask_minus = mask_minus.cuda()
            add_mask = torch.stack((mask_minus, mask_minus, mask_add, mask_add), dim=3)
            add_mask = add_mask.expand(layer_height, layer_width, self.num_priors, -1)

            # box_dim
            box_width = []
            box_height = []
            # 1st box
            box_width.append(float(self.min_size))
            box_height.append(float(self.min_size))
            # 2nd box
            if self.max_size is not None:
                box_width.append(math.sqrt(self.min_size * self.max_size))
                box_height.append(math.sqrt(self.min_size * self.max_size))
            # rest boxes
            for r in self.aspect_ratios:
                box_width.append(self.min_size * math.sqrt(r))
                box_height.append(self.min_size / math.sqrt(r))
            box_width = torch.tensor(box_width).view(1, 1, -1)
            box_height = torch.tensor(box_height).view(1, 1, -1)
            box_dim = torch.stack((box_width, box_height, box_width, box_height), dim=3)
            box_dim = box_dim.expand(layer_height, layer_width, -1, -1)
            if self.use_cuda:
                box_dim = box_dim.cuda()

            # img_dim
            img_w = torch.tensor(img_width).view(1, 1, 1).float()
            img_h = torch.tensor(img_height).view(1, 1, 1).float()
            if self.use_cuda:
                img_w = img_w.cuda()
                img_h = img_h.cuda()
            img_dim = torch.stack((img_w, img_h, img_w, img_h), dim=3)
            img_dim = img_dim.expand(layer_height, layer_width, self.num_priors, -1)

            self.priors = (center_coord + add_mask * box_dim / 2.0) / img_dim
            if self.clip:
                self.priors.clamp_(0.0, 1.0)
        
        return self.priors