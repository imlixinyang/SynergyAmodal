import torch 
import torch.nn as nn
import importlib
import numpy as np
import copy
import torch.nn.functional as F

def import_str(string):
    # From https://github.com/CompVis/taming-transformers
    module, cls = string.rsplit(".", 1)
    return getattr(importlib.import_module(module, package=None), cls)


class EMAModel(nn.Module):
    def __init__(self, *models, beta=0.999):
        super().__init__()
        self.beta = beta 
        self.models = nn.ModuleList([copy.deepcopy(model).eval().requires_grad_(False) for model in models])
        
    def update_average(self, *models):
        with torch.no_grad():
            for model_src, model_tgt in zip(models, self.models):
                param_dict_src = dict(model_src.named_parameters())
                for p_name, p_tgt in model_tgt.named_parameters():
                    p_src = param_dict_src[p_name]
                    assert (p_src is not p_tgt)
                    p_tgt.copy_(self.beta * p_tgt + (1. - self.beta) * p_src)

                buffer_dict_src = dict(model_src.named_buffers())
                for p_name, p_tgt in model_tgt.named_buffers():
                    p_src = buffer_dict_src[p_name]
                    assert (p_src is not p_tgt)
                    p_tgt.copy_(self.beta * p_tgt + (1. - self.beta) * p_src)
                    
    def load_ema_params(self, *models):
        for model_src, model_tgt in zip(models, self.models):
            model_src.load_state_dict(model_tgt.state_dict())

def mask_to_bbox(mask):
    mask = (mask == 1)
    if np.all(~mask):
        return [0, 0, 0, 0]
    assert len(mask.shape) == 2
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return [cmin.item(), rmin.item(), cmax.item() + 1 - cmin.item(), rmax.item() + 1 - rmin.item()] # xywh
                  
def move_to(obj, device):
  if torch.is_tensor(obj):
    return obj.to(device)
  elif isinstance(obj, dict):
    res = {}
    for k, v in obj.items():
      res[k] = move_to(v, device)
    return res
  elif isinstance(obj, list):
    res = []
    for v in obj:
      res.append(move_to(v, device))
    return res
  else:
    return obj

def crop_and_resize(image, visible_mask, target_size=512, padding=0.2, min_size=512, min_scale=-1, padding_value=-1):
    if visible_mask.sum() == 0:
        return image
    
    H, W = image.shape[1], image.shape[2]

    all = image
    
    by0, bx0, w, h = mask_to_bbox((visible_mask).float().cpu().numpy())
    cx, cy = bx0 + h // 2, by0 + w // 2
        
    orginal_length = int(max(h, w) * (1 + padding))
    
    if orginal_length > max(H, W):
        orginal_length = max(H, W)
        
    if min_size > 0 and orginal_length < min_size:
        orginal_length = min_size

    if min_scale > 0 and orginal_length < min_scale * min(H, W):
        orginal_length = min_scale * min(H, W)
                     
    bbox = [cx - orginal_length // 2, cx + orginal_length - orginal_length // 2, cy - orginal_length // 2, cy + orginal_length - orginal_length // 2] # xxyy
            
    # adjust bbox
    if bbox[0] < 0 and bbox[1] < H:
        if -bbox[0] < H - bbox[1]:
            bbox[0], bbox[1] = 0, bbox[1] - bbox[0]
        else:     
            bbox[0], bbox[1] = bbox[0] + (H - bbox[1]), H
                    
    if bbox[0] > 0 and bbox[1] > H:
        if bbox[0] < bbox[1] - H:
            bbox[0], bbox[1] = 0, bbox[1] - bbox[0]
        else:     
            bbox[0], bbox[1] = bbox[0] + (H - bbox[1]), H
                    
    if bbox[2] < 0 and bbox[3] < W:
        if -bbox[2] < W - bbox[3]:
            bbox[2], bbox[3] = 0, bbox[3] - bbox[2]
        else:     
            bbox[2], bbox[3] = bbox[2] + (W - bbox[3]), W
                    
    if bbox[2] > 0 and bbox[3] > W:
        if bbox[2] < bbox[3] - W:
            bbox[2], bbox[3] = 0, bbox[3] - bbox[2]
        else:     
            bbox[2], bbox[3] = bbox[2] + (W - bbox[3]), W     
    
    all = all[:, (bbox[0] if bbox[0] > 0 else 0): (bbox[1] if bbox[1] < H else H),
                   (bbox[2] if bbox[2] > 0 else 0): (bbox[3] if bbox[3] < W else W)]
    
    all = F.pad(all, (-bbox[2] if bbox[2] < 0 else 0, (bbox[3] - W) if bbox[3] > W else 0,
                      -bbox[0] if bbox[0] < 0 else 0, (bbox[1] - H) if bbox[1] > H else 0), 'constant', padding_value)
    
    all = F.interpolate(all[None], size=target_size, mode='bilinear')[0]
    
    return all, bbox

def put_back(patch, H, W, bbox):    
    orginal_length = bbox[1] - bbox[0]
    
    patch = F.interpolate(patch[None], size=orginal_length, mode='bilinear')[0]
    
    patch = patch[:, (-bbox[0] if bbox[0] < 0 else 0): orginal_length - ((bbox[1] - H) if bbox[1] > H else 0),
                     (-bbox[2] if bbox[2] < 0 else 0): orginal_length - ((bbox[3] - W) if bbox[3] > W else 0)]
    
    patch = F.pad(patch, ((bbox[2] if bbox[2] > 0 else 0), W - (bbox[3] if bbox[3] < W else W),
                          (bbox[0] if bbox[0] > 0 else 0), H - (bbox[1] if bbox[1] < H else H),), 'constant', 0)
    
    return patch
     
            
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Utilities for bounding box manipulation and GIoU.
"""
import torch
from torchvision.ops.boxes import box_area


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area


def masks_to_boxes(masks):
    """Compute the bounding boxes around the provided masks

    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.

    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)

    h, w = masks.shape[-2:]

    y = torch.arange(0, h, dtype=torch.float)
    x = torch.arange(0, w, dtype=torch.float)
    y, x = torch.meshgrid(y, x)

    x_mask = (masks * x.unsqueeze(0))
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    y_mask = (masks * y.unsqueeze(0))
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    return torch.stack([x_min, y_min, x_max, y_max], 1)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, length=0):
        self.length = length
        self.reset()

    def reset(self):
        if self.length > 0:
            self.history = []
        else:
            self.count = 0
            self.sum = 0.0
        self.val = 0.0
        self.avg = 0.0

    def update(self, val):
        if self.length > 0:
            self.history.append(val)
            if len(self.history) > self.length:
                del self.history[0]

            self.val = self.history[-1]
            self.avg = np.mean(self.history)
        else:
            self.val = val
            self.sum += val
            self.count += 1
            self.avg = self.sum / self.count
