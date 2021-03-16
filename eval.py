from data import COCODetection, get_label_map, MEANS, COLORS
from yolact import Yolact
from utils.augmentations import BaseTransform, FastBaseTransform, Resize
from utils.functions import MovingAverage, ProgressBar
from layers.box_utils import jaccard, center_size, mask_iou
from utils import timer
from utils.functions import SavePath
from layers.output_utils import postprocess, undo_image_transformation
import pycocotools

from data import cfg, set_cfg, set_dataset

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import argparse
import time
import random
import cProfile
import pickle
import json
import os
from collections import defaultdict
from pathlib import Path
from collections import OrderedDict
from PIL import Image

import matplotlib.pyplot as plt
import cv2

import pcl

IMG_IDX = 0


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


iou_thresholds = [x / 100 for x in range(50, 100, 5)]
coco_cats = {} # Call prep_coco_cats to fill this
coco_cats_inv = {}
color_cache = defaultdict(lambda: {})


def prep_display(cloud, dets_out, img, h, w, undo_transform=True,  class_color=False, mask_alpha=0.45, fps_str=''):
    """
    Note: If undo_transform=False then im_h and im_w are allowed to be None.
    """
    if undo_transform:
        img_numpy = undo_image_transformation(img, w, h)
        img_gpu = torch.Tensor(img_numpy).cuda()
    else:
        img_gpu = img / 255.0
        h, w, _ = img.shape
    
    with timer.env('Postprocess'):
        save = cfg.rescore_bbox
        cfg.rescore_bbox = True
        
        t = postprocess(dets_out, w, h, visualize_lincomb = False,
                                        crop_masks        = True,
                                        score_threshold   = 0)
        cfg.rescore_bbox = save

        ################################### 点云过滤
        img_mask=t[3].cpu().numpy()
        set_img_flat = set()
        for item in img_mask:
          item_flat=np.flatnonzero(item)
          set_img_flat = set_img_flat.union(set(item_flat))
        set_img_flat_list = list(set_img_flat)
        print(len(set_img_flat_list))
        ##pcl=o3d.io.read_point_cloud('/content/yolact/2.5_1ply/2021_02_05_18_22_49_800_points.ply',remove_nan_points=False)

        pcl_after_filter = cloud.extract(set_img_flat_list)

    with timer.env('Copy'):
        idx = t[1].argsort(0, descending=True)[:60]
        
        if cfg.eval_mask_branch:
            # Masks are drawn on the GPU, so don't copy
            masks = t[3][idx]
        classes, scores, boxes = [x[idx].cpu().numpy() for x in t[:3]]

    num_dets_to_consider = min(60, classes.shape[0]) #display top 60 
    for j in range(num_dets_to_consider):
        if scores[j] < 0:
            num_dets_to_consider = j
            break

    # Quick and dirty lambda for selecting the color for a particular index
    # Also keeps track of a per-gpu color cache for maximum speed
    def get_color(j, on_gpu=None):
        global color_cache
        color_idx = (classes[j] * 5 if class_color else j * 5) % len(COLORS)
        
        if on_gpu is not None and color_idx in color_cache[on_gpu]:
            return color_cache[on_gpu][color_idx]
        else:
            color = COLORS[color_idx]
            if not undo_transform:
                # The image might come in as RGB or BRG, depending
                color = (color[2], color[1], color[0])
            if on_gpu is not None:
                color = torch.Tensor(color).to(on_gpu).float() / 255.
                color_cache[on_gpu][color_idx] = color
            return color

    # First, draw the masks on the GPU where we can do it really fast
    # Beware: very fast but possibly unintelligible mask-drawing code ahead
    # I wish I had access to OpenGL or Vulkan but alas, I guess Pytorch tensor operations will have to suffice
    if  cfg.eval_mask_branch and num_dets_to_consider > 0:
        # After this, mask is of size [num_dets, h, w, 1]
        masks = masks[:num_dets_to_consider, :, :, None]
        
        # Prepare the RGB images for each mask given their color (size [num_dets, h, w, 1])
        colors = torch.cat([get_color(j, on_gpu=img_gpu.device.index).view(1, 1, 1, 3) for j in range(num_dets_to_consider)], dim=0)
        masks_color = masks.repeat(1, 1, 1, 3) * colors * mask_alpha

        # This is 1 everywhere except for 1-mask_alpha where the mask is
        inv_alph_masks = masks * (-mask_alpha) + 1
        
        # I did the math for this on pen and paper. This whole block should be equivalent to:
        #    for j in range(num_dets_to_consider):
        #        img_gpu = img_gpu * inv_alph_masks[j] + masks_color[j]
        masks_color_summand = masks_color[0]
        if num_dets_to_consider > 1:
            inv_alph_cumul = inv_alph_masks[:(num_dets_to_consider-1)].cumprod(dim=0)
            masks_color_cumul = masks_color[1:] * inv_alph_cumul
            masks_color_summand += masks_color_cumul.sum(dim=0)

        img_gpu = img_gpu * inv_alph_masks.prod(dim=0) + masks_color_summand
    
    
    # Then draw the stuff that needs to be done on the cpu
    # Note, make sure this is a uint8 tensor or opencv will not anti alias text for whatever reason
    img_numpy = (img_gpu * 255).byte().cpu().numpy()

       
    if num_dets_to_consider == 0:
        return img_numpy

    for j in reversed(range(num_dets_to_consider)):
        x1, y1, x2, y2 = boxes[j, :]
        color = get_color(j)
        score = scores[j]

        cv2.rectangle(img_numpy, (x1, y1), (x2, y2), color, 1)

        _class = cfg.dataset.class_names[classes[j]]      
        text_str = '%s: %.2f' % (_class, score) 

        font_face = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.6
        font_thickness = 1

        text_w, text_h = cv2.getTextSize(text_str, font_face, font_scale, font_thickness)[0]

        text_pt = (x1, y1 - 3)
        text_color = [255, 255, 255]

        cv2.rectangle(img_numpy, (x1, y1), (x1 + text_w, y1 - text_h - 4), color, -1)
        cv2.putText(img_numpy, text_str, text_pt, font_face, font_scale, text_color, font_thickness, cv2.LINE_AA)
       
    return img_numpy, pcl_after_filter


def prep_coco_cats():
    """ Prepare inverted table for category id lookup given a coco cats object. """
    for coco_cat_id, transformed_cat_id_p1 in get_label_map().items():
        transformed_cat_id = transformed_cat_id_p1 - 1
        coco_cats[transformed_cat_id] = coco_cat_id
        coco_cats_inv[coco_cat_id] = transformed_cat_id


def get_coco_cat(transformed_cat_id):
    """ transformed_cat_id is [0,80) as indices in cfg.dataset.class_names """
    return coco_cats[transformed_cat_id]

def get_transformed_cat(coco_cat_id):
    """ transformed_cat_id is [0,80) as indices in cfg.dataset.class_names """
    return coco_cats_inv[coco_cat_id]


class Detections:
    def __init__(self):
        self.bbox_data = []
        self.mask_data = []

    def add_bbox(self, image_id:int, category_id:int, bbox:list, score:float):
        """ Note that bbox should be a list or tuple of (x1, y1, x2, y2) """
        bbox = [bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]]

        # Round to the nearest 10th to avoid huge file sizes, as COCO suggests
        bbox = [round(float(x)*10)/10 for x in bbox]

        self.bbox_data.append({
            'image_id': int(image_id),
            'category_id': get_coco_cat(int(category_id)),
            'bbox': bbox,
            'score': float(score)
        })

    def add_mask(self, image_id:int, category_id:int, segmentation:np.ndarray, score:float):
        """ The segmentation should be the full mask, the size of the image and with size [h, w]. """
        rle = pycocotools.mask.encode(np.asfortranarray(segmentation.astype(np.uint8)))
        rle['counts'] = rle['counts'].decode('ascii') # json.dump doesn't like bytes strings

        self.mask_data.append({
            'image_id': int(image_id),
            'category_id': get_coco_cat(int(category_id)),
            'segmentation': rle,
            'score': float(score)
        })
    
    def dump(self):
        dump_arguments = [
            (self.bbox_data, 'results/bbox_detections.json'),
            (self.mask_data, 'results/mask_detections.json')
        ]

        for data, path in dump_arguments:
            with open(path, 'w') as f:
                json.dump(data, f)
    
    def dump_web(self):
        """ Dumps it in the format for my web app. Warning: bad code ahead! """
        config_outs = ['preserve_aspect_ratio', 'use_prediction_module',
                        'use_yolo_regressors', 'use_prediction_matching',
                        'train_masks']

        output = {
            'info' : {
                'Config': {key: getattr(cfg, key) for key in config_outs},
            }
        }

        image_ids = list(set([x['image_id'] for x in self.bbox_data]))
        image_ids.sort()
        image_lookup = {_id: idx for idx, _id in enumerate(image_ids)}

        output['images'] = [{'image_id': image_id, 'dets': []} for image_id in image_ids]

        # These should already be sorted by score with the way prep_metrics works.
        for bbox, mask in zip(self.bbox_data, self.mask_data):
            image_obj = output['images'][image_lookup[bbox['image_id']]]
            image_obj['dets'].append({
                'score': bbox['score'],
                'bbox': bbox['bbox'],
                'category': cfg.dataset.class_names[get_transformed_cat(bbox['category_id'])],
                'mask': mask['segmentation'],
            })

        with open(os.path.join(args.web_det_path, '%s.json' % cfg.name), 'w') as f:
            json.dump(output, f)
        

        

def _mask_iou(mask1, mask2, iscrowd=False):
    with timer.env('Mask IoU'):
        ret = mask_iou(mask1, mask2, iscrowd)
    return ret.cpu()

def _bbox_iou(bbox1, bbox2, iscrowd=False):
    with timer.env('BBox IoU'):
        ret = jaccard(bbox1, bbox2, iscrowd)
    return ret.cpu()



class APDataObject:
    """
    Stores all the information necessary to calculate the AP for one IoU and one class.
    Note: I type annotated this because why not.
    """

    def __init__(self):
        self.data_points = []
        self.num_gt_positives = 0

    def push(self, score:float, is_true:bool):
        self.data_points.append((score, is_true))
    
    def add_gt_positives(self, num_positives:int):
        """ Call this once per image. """
        self.num_gt_positives += num_positives

    def is_empty(self) -> bool:
        return len(self.data_points) == 0 and self.num_gt_positives == 0

    def get_ap(self) -> float:
        """ Warning: result not cached. """

        if self.num_gt_positives == 0:
            return 0

        # Sort descending by score
        self.data_points.sort(key=lambda x: -x[0])

        precisions = []
        recalls    = []
        num_true  = 0
        num_false = 0

        # Compute the precision-recall curve. The x axis is recalls and the y axis precisions.
        for datum in self.data_points:
            # datum[1] is whether the detection a true or false positive
            if datum[1]: num_true += 1
            else: num_false += 1
            
            precision = num_true / (num_true + num_false)
            recall    = num_true / self.num_gt_positives

            precisions.append(precision)
            recalls.append(recall)

        # Smooth the curve by computing [max(precisions[i:]) for i in range(len(precisions))]
        # Basically, remove any temporary dips from the curve.
        # At least that's what I think, idk. COCOEval did it so I do too.
        for i in range(len(precisions)-1, 0, -1):
            if precisions[i] > precisions[i-1]:
                precisions[i-1] = precisions[i]

        # Compute the integral of precision(recall) d_recall from recall=0->1 using fixed-length riemann summation with 101 bars.
        y_range = [0] * 101 # idx 0 is recall == 0.0 and idx 100 is recall == 1.00
        x_range = np.array([x / 100 for x in range(101)])
        recalls = np.array(recalls)

        # I realize this is weird, but all it does is find the nearest precision(x) for a given x in x_range.
        # Basically, if the closest recall we have to 0.01 is 0.009 this sets precision(0.01) = precision(0.009).
        # I approximate the integral this way, because that's how COCOEval does it.
        indices = np.searchsorted(recalls, x_range, side='left')
        for bar_idx, precision_idx in enumerate(indices):
            if precision_idx < len(precisions):
                y_range[bar_idx] = precisions[precision_idx]

        # Finally compute the riemann sum to get our integral.
        # avg([precision(x) for x in 0:0.01:1])
        return sum(y_range) / len(y_range)


def eval_image(net, cloud, img, save_flag = False, show_falg = False):
    frame = torch.from_numpy(img).cuda().float()
    #print('frame',len(frame.size())) #1248 line
    batch = FastBaseTransform()(frame.unsqueeze(0))
    preds = net(batch)

    
    img_numpy, pcl_after_filter = prep_display(cloud, preds, frame, None, None, undo_transform=False)
    
    if show_falg:
        img_numpy = img_numpy[:, :, (2, 1, 0)]
        plt.imshow(img_numpy)
        plt.title(save_path)
        plt.show()
    
    if save_flag:
        global IMG_IDX
        save_path = "args_out/" + str(IMG_IDX) + ".png"
        IMG_IDX += 1
        cv2.imwrite(save_path, img_numpy) #分割后的图片是否写出来
    
    print("Done")
    return pcl_after_filter


from multiprocessing.pool import ThreadPool
from queue import Queue

class CustomDataParallel(torch.nn.DataParallel):
    """ A Custom Data Parallel class that properly gathers lists of dictionaries. """
    def gather(self, outputs, output_device):
        # Note that I don't actually want to convert everything to the output_device
        return sum(outputs, [])


def init(weight_path):
    args_config = 'yolact_plus_base' + '_config'
    print('Config not specified. Parsed %s from the file name.\n' % None)
    set_cfg(args_config)

    with torch.no_grad():
        if not os.path.exists('results'):
            os.makedirs('results')

        cudnn.fastest = True
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

        dataset = None        

        print('Loading model...', end='')
        net = Yolact()

        net.load_weights(weight_path)
        net.eval()
        print(' Done.')

        net = net.cuda()
        net.detect.use_fast_nms = True
        net.detect.use_cross_class_nms = False
        cfg.mask_proto_debug = False

        return net


if __name__ == '__main__':
    #cloud = o3d.io.read_point_cloud("input/1.ply", remove_nan_points=False)
    net = init('weight/yolact_plus_base.pth')
    cloud = pcl.load("input/1.ply")
    img = cv2.imread("input/1.png")
    
    with torch.no_grad():
        output = eval_image(net, cloud, img)    
        pcl.save(output, "output.ply")
   
          
        
        

