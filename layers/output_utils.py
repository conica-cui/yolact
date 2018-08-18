""" Contains functions used to sanitize and prepare the output of Yolact. """


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2

from data import cfg, mask_type, MEANS
from utils.augmentations import Resize
from utils.functions import sanitize_coordinates
from utils import timer

def postprocess(det_output, w, h, batch_idx=0, interpolation_mode='bilinear'):
    """
    Postprocesses the output of Yolact on testing mode into a format that makes sense,
    accounting for all the possible configuration settings.

    Args:
        - det_output: The dict that Detect outputs.
        - w: The real with of the image.
        - h: The real height of the image.
        - batch_idx: If you have multiple images for this batch, the image's index in the batch.
        - interpolation_mode: Can be 'nearest' | 'area' | 'bilinear' (see torch.nn.functional.interpolate)

    Returns 4 torch Tensors (in the following order):
        - classes [num_det]: The class idx for each detection.
        - scores  [num_det]: The confidence score for each detection.
        - boxes   [num_det, 4]: The bounding box for each detection in absolute point form.
        - masks   [num_det, h, w]: Full image masks for each detection.
    """
    
    dets = det_output['output'].data[batch_idx, :, :]

    # Select only detections with score > 0
    non_zero_score_mask = dets[:, 1].gt(0.0).expand(dets.size(1), dets.size(0)).t()
    dets = torch.masked_select(dets, non_zero_score_mask).view(-1, dets.size(1))
    
    if dets.size(0) == 0:
        return [torch.Tensor()] * 4 # Warning, this is 4 copies of the same thing

    # This doesn't actually do much most of the time since #detections is usually under 100
    if dets.size(0) > cfg.max_num_detections:
        dets = dets[:cfg.max_num_detections]
    
    # Sort descending by score
    _, sort_idx = dets[:, 1].sort(0, descending=True)
    dets = dets[sort_idx, :]

    # im_w and im_h when it concerns bboxes. This is a workaround hack for preserve_aspect_ratio
    b_w, b_h = (w, h)

    # Undo the padding introduced with preserve_aspect_ratio
    if cfg.preserve_aspect_ratio:
        r_w, r_h = Resize.faster_rcnn_scale(w, h, cfg.min_size, cfg.max_size)

        # Get rid of any detections whose centers are outside the image
        boxes = dets[:, 2:6]
        boxes = center_size(boxes)
        s_w, s_h = (r_w/cfg.max_size, r_h/cfg.max_size)
        not_outside = ((boxes[:, 0] > s_w) + (boxes[:, 1] > s_h)) < 1 # not (a or b)
        dets = dets[not_outside]

        # A hack to scale the bboxes to the right size
        b_w, b_h = (cfg.max_size / r_w * w, cfg.max_size / r_h * h)
    
    # Actually extract everything from dets now
    classes = dets[:, 0].int()
    boxes = dets[:, 2:6]
    x1, x2 = sanitize_coordinates(boxes[:, 0], boxes[:, 2], b_w, cast=True)
    y1, y2 = sanitize_coordinates(boxes[:, 1], boxes[:, 3], b_h, cast=True)
    boxes = torch.stack((x1, y1, x2, y2), dim=1)
    scores = dets[:, 1]
    masks = dets[:, 6:]

    if cfg.mask_type == mask_type.lincomb:
        # At this points masks is only the coefficients
        proto_data = det_output['proto_data'][batch_idx]
        
        masks = torch.matmul(proto_data, masks.t())
    
        # Permute into the correct output shape [num_dets, proto_h, proto_w]
        masks = masks.permute(2, 0, 1).contiguous()
    
        if cfg.mask_proto_second_nonlinearity == 'sigmoid':
            masks = torch.sigmoid(masks)
        elif cfg.mask_proto_second_nonlinearity == 'relu':
            masks = F.relu(masks, inplace=True)

        # Scale masks up to the full image
        if cfg.preserve_aspect_ratio:
            # Undo padding
            masks = masks[:, :int(r_h/cfg.max_size*proto_data.size(1)), :int(r_w/cfg.max_size*proto_data.size(2))]
        masks = F.interpolate(masks.unsqueeze(0), (h, w), mode=interpolation_mode, align_corners=False).squeeze(0)

        # "Crop" predicted masks by zeroing out everything not in the predicted bbox
        # TODO: Write a cuda implementation of this to get rid of the loop
        num_dets = boxes.size(0)
        crop_mask = torch.zeros(num_dets, h, w, device=masks.device)
        for jdx in range(num_dets):
            crop_mask[jdx, y1[jdx]:y2[jdx], x1[jdx]:x2[jdx]] = 1
        masks = masks * crop_mask

        # Binarize the masks
        masks = masks.gt(0.5).float()


    elif cfg.mask_type == mask_type.direct:
        # Upscale masks
        full_masks = torch.zeros(masks.size(0), h, w)

        for jdx in range(masks.size(0)):
            x1, y1, x2, y2 = boxes[jdx, :]

            mask_w = x2 - x1
            mask_h = y2 - y1

            # Just in case
            if mask_w * mask_h <= 0 or mask_w < 0:
                continue
            
            mask = masks[jdx, :].view(1, 1, cfg.mask_size, cfg.mask_size)
            mask = F.interpolate(mask, (mask_h, mask_w), mode=interpolation_mode, align_corners=False)
            mask = mask.gt(0.5).float()
            full_masks[jdx, y1:y2, x1:x2] = mask
        
        masks = full_masks

    return classes, scores, boxes, masks


    


def undo_image_transformation(img, w, h):
    """
    Takes a transformed image tensor and returns a numpy ndarray that is untransformed.
    Arguments w and h are the original height and width of the image.
    """
    img_numpy = (img.permute(1, 2, 0).cpu().numpy() / 255.0 + np.array(MEANS) / 255.0).astype(np.float32)
    img_numpy = np.clip(img_numpy, 0, 1)

    if cfg.preserve_aspect_ratio:
        # Undo padding
        r_w, r_h = Resize.faster_rcnn_scale(w, h, cfg.min_size, cfg.max_size)
        img_numpy = img_numpy[:r_h, :r_w]

        # Undo resizing
        img_numpy = cv2.resize(img_numpy, (w,h))

    else:
        return cv2.resize(img_numpy, (w,h))

