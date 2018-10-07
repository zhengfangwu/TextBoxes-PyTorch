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
    # matches: num_priors * 4
    # priors: num_priors * 4
    # variances: 4
    # return: num_priors * 4

    # prior_width = priors[:, :, 2] - priors[:, :, 0]
    # prior_height = priors[:, :, 3] - priors[:, :, 1]
    prior_wh = priors[:, 2:] - priors[:, :2]
    # prior_center_x = (priors[:, :, 0] + priors[:, :, 2]) / 2.0
    # prior_center_y = (priors[:, :, 1] + priors[:, :, 3]) / 2.0
    prior_cxy = (priors[:, :2] + priors[:, 2:]) / 2.0

    # bbox_width = matches[:, :, 2] - matches[:, :, 0]
    # bbox_height = matches[:, :, 3] - matches[:, :, 1]
    bbox_wh = matches[:, 2:] - matches[:, :2]
    # bbox_center_x = (matches[:, :, 0] + matches[:, :, 2]) / 2.0
    # bbox_center_y = (matches[:, :, 1] + matches[:, :, 3]) / 2.0
    bbox_cxy = (matches[:, :2] + matches[:, 2:]) / 2.0

    encode_bbox = []
    # encode_bbox.append((bbox_center_x - prior_center_x) / (prior_width * variances[0]))
    # encode_bbox.append((bbox_center_y - prior_center_y) / (prior_height * variances[1]))
    encode_bbox.append((bbox_cxy - prior_cxy) / (prior_wh * variances[:2]))
    # encode_bbox.append((torch.log(bbox_width / prior_width) / variances[2]))
    # encode_bbox.append((torch.log(bbox_height / prior_height) / variances[3]))
    encode_bbox.append(torch.log(bbox_wh / prior_wh) / variances[2:])

    # return torch.stack(encode_bbox, dim=2)
    return torch.cat(encode_bbox, dim=1)


def match(gt, priors, threshold, variances, use_cuda):
    # Process one sample at a time since num_objects is different.
    # input:
    #   gt: num_objects * 4
    #   priors: num_priors * 4
    #   variances: 4
    # output:
    #   loc: num_priors * 4 (dx, dw, log(per dh), log(per dw))
    #   conf: num_priors
    num_objects = gt.size(0)
    num_priors = priors.size(0)

    overlaps = jaccard(gt, priors) # num_gt * num_priors

    # [num_gt] best prior for each groundtruth
    best_prior_overlap, best_prior_idx = torch.max(overlaps, dim=1, keepdim=False)
    # [num_priors] best object for each prior
    best_truth_overlap, best_truth_idx = torch.max(overlaps, dim=0, keepdim=False)
    
    best_truth_overlap.index_fill_(0, best_prior_idx, 1.1)
    for j in range(best_prior_idx.size(0)):
        best_truth_idx[best_prior_idx[j]] = j
    
    matches = gt[best_truth_idx]
    # num_prior * 4

    conf = torch.ones(num_priors).long()
    if use_cuda:
        conf = conf.cuda()
    conf[best_truth_overlap < threshold] = 0
    loc = encode(matches, priors, variances)
    # print(conf.size(), loc.size())
    return loc, conf


def log_sum_exp(x):
    x_max = x.max()
    return torch.log(torch.sum(torch.exp(x - x_max), dim=2, keepdim=False)) + x_max

def init_weights(m):
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.zero_()

def init_conv_layer(layer_a, layer_b):
    layer_a.weight = layer_b.weight
    layer_a.bias = layer_b.bias

def initialize(net):
    net.apply(init_weights)
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
    print('test ok')

def save_checkpoint(epoch_idx, checkpoint_folder, net, optimizer):
    """ A comprehensive checkpoint saving function
    """
    checkpoint = {
        'epoch': epoch_idx,
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    checkpoint_path = os.path.join(checkpoint_folder, 'checkpoint_epoch' + str(epoch_idx))
    torch.save(checkpoint, checkpoint_path)

def load_checkpoint(checkpoint_path, net, optimizer):
    """ A comprehensive checkpoint loading function
    """
    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch']