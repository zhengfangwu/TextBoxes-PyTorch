import torch
import os
import torchvision

def jaccard(box_a, box_b):
    # box_a: num_a * 4
    # box_b: num_b * 4
    # 4 = (xmin, ymin, xmax, ymax)
    num_a = box_a.size(0)
    num_b = box_b.size(0)
    min_xy_max = torch.min(box_a[:, 2:].unsqueeze(1).expand(num_a, num_b, 2),
                           box_b[:, 2:].unsqueeze(0).expand(num_a, num_b, 2))
    max_xy_min = torch.max(box_a[:, :2].unsqueeze(1).expand(num_a, num_b, 2),
                           box_b[:, :2].unsqueeze(0).expand(num_a, num_b, 2))
    interval = torch.clamp(min_xy_max - max_xy_min, min=0)
    intersec = interval[:, :, 0] * interval[:, :, 1]

    area_a = ((box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1])).\
            unsqueeze(1).expand(num_a, num_b)
    area_b = ((box_b[:, 2] - box_b[:, 0]) * (box_b[:, 3] - box_b[:, 1])).\
            unsqueeze(0).expand(num_a, num_b)

    return intersec / (area_a + area_b - intersec)

def encode(matches, priors, variances):
    # matches: num_priors * 4 (xmin, ymin, xmax, ymax)
    # priors: num_priors * 4 (xmin, ymin, xmax, ymax)
    # variances: 4 (to align scale between offset and coordinates)
    # return: num_priors * 4

    prior_wh = priors[:, 2:] - priors[:, :2]
    prior_cxy = (priors[:, :2] + priors[:, 2:]) / 2.0

    bbox_wh = matches[:, 2:] - matches[:, :2]
    bbox_cxy = (matches[:, :2] + matches[:, 2:]) / 2.0

    encode_bbox = []
    encode_bbox.append((bbox_cxy - prior_cxy) / (prior_wh * variances[:2]))
    encode_bbox.append(torch.log(bbox_wh / prior_wh) / variances[2:])
    
    return torch.cat(encode_bbox, dim=1)


def decode(loc, priors, variances):
    """
    Input:
        loc: num_priors * 4
        priors: num_priors * 4
        variances: 4
    Return:
        decode_bbox: num_priors * 4 (xmin, ymin, xmax, ymax)
    """
    prior_wh = priors[:, 2:] - priors[:, :2]
    prior_cxy = (priors[:, :2] + priors[:, 2:]) / 2.0

    bbox_cxy = prior_cxy + loc[:, :2] * variances[:2] * prior_wh
    bbox_wh = prior_wh * torch.exp(loc[:, 2:] * variances[2:])

    decode_bbox = torch.cat([bbox_cxy, bbox_wh], dim=1)
    decode_bbox[:, :2] -= decode_bbox[:, 2:] / 2
    decode_bbox[:, 2:] += decode_bbox[:, :2]

    return decode_bbox


def match(gt, priors, threshold, variances, device):
    # Process one sample at a time since num_objects is different.
    # input:
    #   gt: num_objects * 4
    #   priors: num_priors * 4
    #   variances: 4
    # output:
    #   loc: num_priors * 4 (dx, dw, log(per dh), log(per dw))
    #   conf: num_priors

    # print('gt range', gt.min(), gt.max())
    # print('priors range', priors.min(), priors.max())

    num_objects = gt.size(0)
    num_priors = priors.size(0)

    overlaps = jaccard(gt, priors) # num_gt * num_priors
    # print('overlaps min:', overlaps.min(), ' max:', overlaps.max())

    # [num_gt] best prior for each groundtruth
    best_prior_overlap, best_prior_idx = torch.max(overlaps, dim=1, keepdim=False)
    # [num_priors] best object for each prior
    best_truth_overlap, best_truth_idx = torch.max(overlaps, dim=0, keepdim=False)
    
    # For each groundtruth box, match the best matching prior box
    best_truth_overlap.index_fill_(0, best_prior_idx, 2)
    for j in range(best_prior_idx.size(0)):
        best_truth_idx[best_prior_idx[j]] = j
    
    matches = gt[best_truth_idx]
    # num_prior * 4

    conf = torch.ones(num_priors).long()
    conf = conf.to(device)
    conf[best_truth_overlap < threshold] = 0
    loc = encode(matches, priors, variances)
    # print(conf.size(), loc.size())
    return loc, conf


def log_sum_exp(x):
    x_max = x.max()
    return torch.log(torch.sum(torch.exp(x - x_max), dim=1, keepdim=True)) + x_max

def init_weights(m):
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.zero_()

def init_conv_layer(layer_a, layer_b):
    # layer_a.weight.copy_(layer_b.weight.data.detach())
    # layer_a.bias.cpoy_(layer_b.bias.data.detach())
    layer_a.load_state_dict(layer_b.state_dict())

def freeze_conv_layer(layer):
    layer.weight.requires_grad = False
    layer.bias.requires_grad = False

def initialize(net, load_vgg):
    net.apply(init_weights)

    if load_vgg:
        vgg = torchvision.models.vgg16(pretrained=True)
        vgg = vgg.features

        init_conv_layer(net.conv1_1, vgg[0])
        init_conv_layer(net.conv1_2, vgg[2])
        init_conv_layer(net.conv2_1, vgg[5])
        init_conv_layer(net.conv2_2, vgg[7])
        init_conv_layer(net.conv3_1, vgg[10])
        init_conv_layer(net.conv3_2, vgg[12])
        init_conv_layer(net.conv3_3, vgg[14])
        init_conv_layer(net.conv4_1, vgg[17])
        init_conv_layer(net.conv4_2, vgg[19])
        init_conv_layer(net.conv4_3, vgg[21])
        init_conv_layer(net.conv5_1, vgg[24])
        init_conv_layer(net.conv5_2, vgg[26])
        init_conv_layer(net.conv5_3, vgg[28])

        # freeze_conv_layer(net.conv1_1)
        # freeze_conv_layer(net.conv1_2)
        # freeze_conv_layer(net.conv2_1)
        # freeze_conv_layer(net.conv2_2)

def save_checkpoint(epoch_idx, checkpoint_folder, net, optimizer):
    """ A comprehensive checkpoint saving function
    """
    checkpoint = {
        'epoch': epoch_idx,
        'model_state_dict': net.state_dict(),
        # 'optimizer_state_dict': optimizer.state_dict()
    }
    checkpoint_path = os.path.join(checkpoint_folder, 'checkpoint_epoch' + str(epoch_idx))
    torch.save(checkpoint, checkpoint_path)

def load_checkpoint(checkpoint_path, net, optimizer):
    """ A comprehensive checkpoint loading function
    """
    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch']

def nms(bbox, scores, overlap_threshold, topk):
    """
    Input:
        bbox: num_priors x 4
        scores: num_priors
        overlap_threshold: threshold for IoU score
        topk: maximum numver of box predictions to consider
    """
    keep = scores.new(scores.size(0)).zero_().long()
    if bbox.numel() == 0:
        return keep
    
    x1 = bbox[:, 0]
    y1 = bbox[:, 1]
    x2 = bbox[:, 2]
    y2 = bbox[:, 3]
    area = (x2 - x1) * (y2 - y1)
    y, idx = scores.sort(0) # ascending
    idx = idx[-topk:]
    
    xx1 = bbox.new()
    yy1 = bbox.new()
    xx2 = bbox.new()
    yy2 = bbox.new()
    w = bbox.new()
    h = bbox.new()

    count = 0
    while idx.numel() > 0:
        i = idx[-1]
        keep[count] = i
        count += 1
        if idx.size(0) == 1:
            break
        idx = idx[:-1]
        torch.index_select(x1, 0, idx, out=xx1)
        torch.index_select(y1, 0, idx, out=yy1)
        torch.index_select(x2, 0, idx, out=xx2)
        torch.index_select(y2, 0, idx, out=yy2)
        
        xx1 = torch.clamp(xx1, min=x1[i])
        yy1 = torch.clamp(yy1, min=y1[i])
        xx2 = torch.clamp(xx2, max=x2[i])
        yy2 = torch.clamp(yy2, max=y2[i])

        w.resize_as_(xx2)
        h.resize_as_(yy2)

        w = (xx2 - xx1).clamp(min = 0.0)
        h = (yy2 - yy1).clamp(min = 0.0)
        
        inter = w * h
        remain_areas = torch.index_select(area, 0, idx)
        union = (remain_areas - inter) + area[i]
        IoU = inter / union
        idx = idx[IoU.le(overlap_threshold)]
        
    return keep, count