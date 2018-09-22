import torch

def jaccard(box_a, box_b):
    # box_a: batch_size * num_a * 4
    # box_b: batch_size * num_b * 4
    # 4 = (xmin, ymin, xmax, ymax)
    assert box_a.size(0) == box_b.size(0), "BoxA and BoxB should have the same batch size"
    batch_size = box_a.size(0)
    num_a = box_a.size(1)
    num_b = box_b.size(1)
    min_xy_max = torch.min(box_a[:, :, 2:].unsqueeze(2).expand(batch_size, num_a, num_b, 2),
                           box_b[:, :, 2:].unsqueeze(1).expand(batch_size, num_a, num_b, 2))
    max_xy_min = torch.max(box_a[:, :, :2].unsqueeze(2).expand(batch_size, num_a, num_b, 2),
                           box_b[:, :, :2].unsqueeze(1).expand(batch_size, num_a, num_b, 2))
    interval = torch.clamp(min_xy_max - max_xy_min, min=0)
    intersec = interval[:, :, :, 0] * interval[:, :, :, 1]

    area_a = ((box_a[:, :, 2] - box_a[:, :, 0]) * (box_a[:, :, 3] - box_a[:, :, 1])).\
            unsqueeze(2).expand_as(batch_size, num_a, num_b)
    area_b = ((box_b[:, :, 2] - box_b[:, :, 0]) * (box_b[:, :, 3] - box_a[:, :, 1])).\
            unsqueeze(1).expand_as(batch_size, num_a, num_b)

    return intersec / (area_a + area_b - intersec)

def encode(matches, priors, variances):
    # matches: batch_size * num_priors * 4
    # priors: batch_size * num_priors * 4
    # variances: 4
    # return: batch_size * num_priors * 4

    # prior_width = priors[:, :, 2] - priors[:, :, 0]
    # prior_height = priors[:, :, 3] - priors[:, :, 1]
    prior_wh = priors[:, :, 2:] - priors[:, :, :2]
    # prior_center_x = (priors[:, :, 0] + priors[:, :, 2]) / 2.0
    # prior_center_y = (priors[:, :, 1] + priors[:, :, 3]) / 2.0
    prior_cxy = (priors[:, :, :2] + priors[:, :, 2:]) / 2.0

    # bbox_width = matches[:, :, 2] - matches[:, :, 0]
    # bbox_height = matches[:, :, 3] - matches[:, :, 1]
    bbox_wh = matches[:, :, 2:] - matches[:, :, :2]
    # bbox_center_x = (matches[:, :, 0] + matches[:, :, 2]) / 2.0
    # bbox_center_y = (matches[:, :, 1] + matches[:, :, 3]) / 2.0
    bbox_cxy = (matches[:, :, :2] + matches[:, :, 2:]) / 2.0

    encode_bbox = []
    # encode_bbox.append((bbox_center_x - prior_center_x) / (prior_width * variances[0]))
    # encode_bbox.append((bbox_center_y - prior_center_y) / (prior_height * variances[1]))
    encode_bbox.append((bbox_cxy - prior_cxy) / (prior_wh * variances[:2]))
    # encode_bbox.append((torch.log(bbox_width / prior_width) / variances[2]))
    # encode_bbox.append((torch.log(bbox_height / prior_height) / variances[3]))
    encode_bbox.append(torch.log(bbox_wh / prior_wh) / variances[2:])

    # return torch.stack(encode_bbox, dim=2)
    return torch.cat(encode_bbox, dim=2)


def match_bbox(gt, priors, threshold, variances, use_cuda):
    # gt: batch_size * num_objects * 4
    # priors: batch_size * num_priors * 4
    # variances: 4
    assert gt.size(0) == priors.size(0), "Ground-truth and prior boxes should have the same batch size"
    batch_size = gt.size(0)
    overlaps = jaccard(gt, priors) # num_gt * num_priors

    # [batch_size * num_gt] best prior for each groundtruth
    best_prior_overlap, best_prior_idx = torch.max(overlaps, dim=2, keepdim=False)
    # [batch_size * num_priors] best object for each prior
    best_truth_overlap, best_truth_idx = torch.max(overlaps, dim=1, keepdim=False)

    matches = []
    for i in range(batch_size):
        best_truth_overlap[i].index_fill_(0, best_prior_idx[i], 1.1)
        for j in range(best_prior_idx.size(1)):
            best_truth_idx[i, best_prior_idx[i][j]] = j
    
        matches.append(gt[i, best_truth_idx[i]])
    matches = torch.stack(matches, dim=0)
    # batch_size * num_prior * 4

    conf = torch.ones(gt.size(1))[best_truth_idx].unsqueeze(0).expand(batch_size, gt.size(1))
    if use_cuda:
        conf = conf.cuda()
    conf[best_truth_overlap < threshold] = 0
    loc = encode(matches, priors, variances)
    return loc, conf