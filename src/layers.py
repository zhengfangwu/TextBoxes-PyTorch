import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import match, log_sum_exp, nms, decode

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
    def __init__(self, min_size, max_size, aspect_ratios, clip, device, round_up_bbox):
        super(PriorBoxLayer, self).__init__()

        self.min_size = min_size
        self.max_size = max_size
        self.aspect_ratios = aspect_ratios
        self.clip = clip
        self.device = device
        self.round_up_bbox = round_up_bbox # TODO

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
        if self.priors is None:

            # center_coord
            w = torch.arange(0, layer_width).view(1, -1).expand(layer_height, layer_width).float()
            h = torch.arange(0, layer_height).view(-1, 1).expand(layer_height, layer_width).float()
            center_x = (w + 0.5) * step_x
            center_y = (h + 0.5) * step_y
            center_coord = torch.stack((center_x, center_y, center_x, center_y), dim=2)
            center_coord = center_coord.unsqueeze(2).expand(-1, -1, self.num_priors, -1)

            # add_mask
            mask_add = torch.tensor(1.0).view(1, 1, 1)
            mask_minus = torch.tensor(-1.0).view(1, 1, 1)
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

            # img_dim
            img_w = torch.tensor(img_width).view(1, 1, 1).float()
            img_h = torch.tensor(img_height).view(1, 1, 1).float()
            img_dim = torch.stack((img_w, img_h, img_w, img_h), dim=3)
            img_dim = img_dim.expand(layer_height, layer_width, self.num_priors, -1)

            self.priors = (center_coord + add_mask * box_dim / 2.0) / img_dim
            if self.clip:
                self.priors.clamp_(0.0, 1.0)

            self.priors = self.priors.to(self.device)
        
        return self.priors

if __name__ == "__main__":

    # test PriorBoxLayer
    prior_box = PriorBoxLayer(5, 10, [2, 3], True, torch.device('cuda:0'), False)
    img = torch.randn(1, 3, 30, 30)
    feature = torch.randn(1, 5, 4 , 4)
    priors = prior_box.forward(img, feature)
    torch.set_printoptions(edgeitems=5)
    print(priors.size())
    print(priors)


class MultiBoxLoss(nn.Module):

    def __init__(self, threshold, variances, neg_ratio, device):
        super(MultiBoxLoss, self).__init__()

        self.threshold = threshold
        self.variances = variances
        self.neg_ratio = neg_ratio
        self.device = device

        self.num_classes = 2

    def forward(self, predictions, targets):
        # predictions:
        #   loc_data: batch_size * num_priors * 4
        #   conf_data: batch_size * num_priors * num_classes (2)
        #   priors: num_priors * 4
        # targets:
        #   list, len=batch_size, each (num_objects, 4), no labels

        loc_data, conf_data, priors = predictions

        # ################################
        # import matplotlib.pyplot as plt
        # import numpy as np
        # import cv2
        # for prior in priors:
        #     tmp_img = np.zeros((300, 300, 3), dtype=np.uint8)
        #     xmin = int(prior[0].item() * 300)
        #     ymin = int(prior[1].item() * 300)
        #     xmax = int(prior[2].item() * 300)
        #     ymax = int(prior[3].item() * 300)
        #     cv2.rectangle(tmp_img, (xmin, ymin), (xmax, ymax), (255, 0, 0))
        #     print('(%d, %d) --> (%d, %d)' % (xmin, ymin, xmax, ymax))
        #     # plt.imshow(tmp_img)
        #     # plt.show()
        #     input()

        # print(len(targets))
        # print('==== priors: min:', priors.min(), 'max:', priors.max())
        # print('==== targets: ', end='')
        # for target in targets:
        #     print('min:', target.min(), 'max:', target.max(), end='')
        # print('')

        batch_size = loc_data.size(0)
        num_priors = loc_data.size(1)
        # dim check
        assert conf_data.size(0) == loc_data.size(0), "conf and loc should have same batch size"
        assert conf_data.size(1) == loc_data.size(1), "conf and loc should have same amount of prior boxes"

        loc_t = []
        conf_t = []
        for target in targets:
            _loc, _conf = match(target, priors, self.threshold, self.variances, self.device)
            loc_t.append(_loc)
            conf_t.append(_conf)
        loc_t = torch.stack(loc_t, dim=0) # batch_size * num_priors * 4
        # print(loc_t.max())
        conf_t = torch.stack(conf_t, dim=0) # batch_size * num_priors
        # print('loc_t, conf_t', loc_t.size(), conf_t.size())

        pos = conf_t > 0 # batch_size * num_priors

        # Localization loss
        # print('pos', pos.size())
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        # print('pos_idx', pos_idx.size())
        loc_p = loc_data[pos_idx].view(-1, 4)
        # print('=========== loc_p max', loc_p.max())
        loc_t = loc_t[pos_idx].view(-1, 4)
        # print('=========== loc_t max', loc_t.max())
        loss_loc = F.smooth_l1_loss(loc_p, loc_t, size_average=False)

        batch_conf = conf_data.view(-1, self.num_classes)
        loss_c = (log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))).view(batch_size, -1)

        # Hard negative mining
        loss_c[pos] = 0 # batch_size * num_priors
        loss_c = loss_c.view(batch_size, -1)
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1) # batch_size * num_priors

        num_pos = pos.long().sum(1, keepdim=True) # [batch_size]
        num_neg = torch.clamp(self.neg_ratio * num_pos, max=pos.size(1) - 1) # [batch_size]
        neg = idx_rank < num_neg.expand_as(idx_rank) # batch_size * num_priors

        # Confidence loss including positive and negative examples
        pos_idx = pos.unsqueeze(2).expand(batch_size, num_priors, self.num_classes)
        neg_idx = neg.unsqueeze(2).expand(batch_size, num_priors, self.num_classes)
        conf_p = conf_data[(pos_idx + neg_idx).gt(0)].view(-1, self.num_classes)
        targets_weighted = conf_t[(pos + neg).gt(0)]
        loss_conf = F.cross_entropy(conf_p, targets_weighted)

        N = num_pos.sum().item()
        loss_loc /= N
        loss_conf /= N

        return loss_loc, loss_conf


class DetectionOutput(nn.Module):

    def __init__(self, conf_threshold, nms_threshold, variances, topk):
        super(DetectionOutput, self).__init__()

        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.variances = variances
        self.topk = topk
        
        self.num_classes = 2
        self.bkg_label = 0

    def forward(self, loc_data, conf_data, priors):
        """
        Input:
            loc_data: batch x num_priors x 4 (dx, dy, log_dw, log_dh)
            conf_data: batch x num_priors x num_classes (2)
            priors: num_priors x 4 (xmin, ymin, xmax, ymax)
        """
        batch_size = loc_data.size(0)
        num_priors = priors.size(0)
        assert loc_data.size(1) == num_priors, "loc_data should have the same number of priors"
        assert conf_data.size(1) == num_priors, "conf_data should have the same number of priors"

        conf_pred = conf_data.transpose(2, 1) # batch x num_classes (2) x num_priors

        output = torch.zeros(batch_size, self.num_classes, self.topk, 5)
        # print('output', output.size())

        for i in range(batch_size):
            # print(loc[1, :2].max())
            decode_bbox = decode(loc_data[i], priors, self.variances) # num_priors * 4 (xmin, ymin, xmax, ymax)
            conf_scores = conf_pred[i].clone()
            # print(decode_bbox.max())
            # TODO: decode_bbox larger than 1
            
            # TextBoxes localization only have two labels (background and text)
            conf_mask = conf_scores[1].gt(self.conf_threshold) # num_priors
            scores = conf_scores[1][conf_mask]
            if scores.size(0) == 0:
                continue
            bbox_mask = conf_mask.unsqueeze(1).expand_as(decode_bbox) # num_priors * 4
            bbox = decode_bbox[bbox_mask].view(-1, 4)
            idx, count = nms(bbox, scores, self.nms_threshold, self.topk)

            output[i, 1, :count] = torch.cat((scores[idx[:count]].unsqueeze(1), \
                                    bbox[idx[:count]]), 1)
        
        # TODO: error here, to be fixed
        flat = output.view(batch_size, -1, 5) # batch x (2*topk) x 5
        # print('flat', flat.size()) # 1 x 10 x 5
        _, idx = flat[:, :, 0].sort(1, descending=True)
        _, rank = idx.sort(1)
        # print('rank', rank.size()) # 1 x 10
        # print((flat == 0).sum(), end=' ')
        # print((rank < self.topk).sum())
        flat[(rank < self.topk).unsqueeze(-1).expand_as(flat)].fill_(0)
        # TODO: seems no value is changed

        # print((flat == 0).sum())
        # output = flat[(rank < self.topk).unsqueeze(-1).expand_as(flat)].fill_(0)
        # print(output.size()) # 1000
        # output = output.view(batch_size, self.num_classes, self.topk, 5)
        return output