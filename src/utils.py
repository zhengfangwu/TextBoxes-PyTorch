import torch

def jaccard(box_a, box_b):
    # Suppose inputs are (num_priors * 4)
    # 4 = (xmin, ymin, xmax, ymax)
    num_prior_a = box_a.size()
    num_prior_b = box_b.size()
    min_xy_max = torch.min(box_a[:, 2:].unsqueeze(1).expand(num_prior_a, num_prior_b, 2),
                           box_b[:, 2:].unsqueeze(0).expand(num_prior_a, num_prior_b, 2))
    max_xy_min = torch.max(box_a[:, :2].unsqueeze(1).expand(num_prior_a, num_prior_b, 2),
                           box_b[:, :2].unsqueeze(0).expand(num_prior_a, num_prior_b, 2))
    interval = torch.clamp(min_xy_max - max_xy_min, min=0)
    intersec = interval[:, :, 0] * interval[:, :, 1]

    area_a = ((box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1])).\
            view(-1, 1).expand_as(num_prior_a, num_prior_b)
    area_b = ((box_b[:, 2] - box_b[:, 0]) * (box_b[:, 3] - box_a[:, 1])).\
            view(-1, 1).expand_as(num_prior_a, num_prior_b)

    return intersec / (area_a + area_b - intersec)

def match_bbox(gt, priors):
    overlaps = jaccard(gt, priors) # num_gt * num_priors

    # [num_objects] best prior for each groundtruth
    best_prior_overlap, best_prior_idx = torch.max(overlaps, dim=1, keepdim=False)
    # [num_priors] best object for each prior
    best_truth_overlap, best_truth_idx = torch.max(overlaps, dim=0, keepdim=False)