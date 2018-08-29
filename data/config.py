# config.py
import os.path
import copy

# gets home dir cross platform
HOME = os.path.expanduser("~")

# for making bounding boxes pretty
COLORS = ((255, 0, 0, 128), (0, 255, 0, 128), (0, 0, 255, 128),
          (0, 255, 255, 128), (255, 0, 255, 128), (255, 255, 0, 128))

MEANS = (104, 117, 123)


class Config(object):
    """
    Holds the config for various networks.
    To get the currently active config, call get_cfg().

    To use, just do cfg.x instead of cfg['x'].
    """

    def __init__(self, config_dict):
        for key, val in config_dict.items():
            self.__setattr__(key, val)

    def copy(self, new_config_dict={}):
        """
        Copies this config into a new config object, making
        the changes given by new_config_dict.
        """

        ret = Config(vars(self))
        
        for key, val in new_config_dict.items():
            ret.__setattr__(key, val)

        return ret

    def replace(self, new_config_dict):
        """
        Copies new_config_dict into this config object.
        Note: new_config_dict can also be a config object.
        """
        if isinstance(new_config_dict, Config):
            new_config_dict = vars(new_config_dict)

        for key, val in new_config_dict.items():
            self.__setattr__(key, val)


# Datasets
coco_dataset = Config({
    'name': 'COCO',
    'train': 'train2014',
    'valid': 'val2014'
})

# Backbones
from backbone import ResNetBackbone, VGGBackbone
from torchvision.models.vgg import cfg as vggcfg
from math import sqrt
import torch

resnet101_backbone = Config({
    'name': 'ResNet101',
    'path': 'resnet101_reducedfc.pth',
    'type': ResNetBackbone,
    'args': ([3, 4, 23, 3],),

    'selected_layers': list(range(2, 8)),
    'pred_scales': [[1]]*6,
    'pred_aspect_ratios': [ [[0.66685089, 1.7073535, 0.87508774, 1.16524493, 0.49059086]] ] * 6,
})

resnet50_backbone = resnet101_backbone.copy({
    'name': 'ResNet50',
    'path': 'resnet50-19c8e357.pth',
    'type': ResNetBackbone,
    'args': ([3, 4, 6, 3],),
})

vgg16_arch = [[64, 64],
              [ 'M', 128, 128],
              [ 'M', 256, 256, 256],
              [('M', {'kernel_size': 2, 'stride': 2, 'ceil_mode': True}), 512, 512, 512],
              [ 'M', 512, 512, 512],
              [('M',  {'kernel_size': 3, 'stride':  1, 'padding':  1}),
               (1024, {'kernel_size': 3, 'padding': 6, 'dilation': 6}),
               (1024, {'kernel_size': 1})]]

vgg16_backbone = Config({
    'name': 'VGG16',
    'path': 'vgg16_reducedfc.pth',
    'type': VGGBackbone,
    'args': (vgg16_arch, [(256, 2), (128, 2), (128, 1), (128, 1)], [3]),

    'selected_layers': [3] + list(range(5, 10)),
    'pred_scales': [[5, 4]]*6,
    'pred_aspect_ratios': [ [[1], [1, sqrt(2), 1/sqrt(2), sqrt(3), 1/sqrt(3)][:n]] for n in [3, 5, 5, 5, 3, 3] ],
})

mask_type = Config({
    # Direct produces masks directly as the output of each pred module.
    # Parameters: mask_size, use_gt_bboxes
    'direct': 0,

    # Lincomb produces coefficients as the output of each pred module then uses those coefficients
    # to linearly combine features from an earlier convout to create image-sized masks.
    # Parameters:
    #   - masks_to_train (int): Since we're producing (near) full image masks, it'd take too much
    #                           vram to backprop on every single mask. Thus we select only a subset.
    #   - mask_proto_src (int): The input layer to the mask prototype generation network. This is an
    #                           index in backbone.layers. Use to use the image itself instead.
    #   - mask_proto_net (list<tuple>): A list of layers in the mask proto network with the last one
    #                                   being where the masks are taken from. Each conv layer is in
    #                                   the form (num_features, kernel_size, **kwdargs). An empty
    #                                   list means to use the source for prototype masks. If the
    #                                   kernel_size is negative, this creates a deconv layer instead.
    #   - mask_proto_bias (bool): Whether to include an extra coefficient that corresponds to a proto
    #                             mask of all ones.
    #   - mask_proto_prototype_activation (func): The activation to apply to each prototype mask.
    #   - mask_proto_mask_activation (func): After summing the prototype masks with the predicted
    #                                        coeffs, what activation to apply to the final mask.
    #   - mask_proto_coeff_activation (func): The activation to apply to the mask coefficients.
    #   - mask_proto_crop (bool): If True, crop the mask with the predicted bbox during training.
    #   - mask_proto_loss (str [l1|disj]): If not None, apply an l1 or disjunctive regularization
    #                                      loss directly to the prototype masks.
    'lincomb': 1,
})

# Self explanitory. For use with mask_proto_*_activation
activation_func = Config({
    'tanh': torch.tanh,
    'sigmoid': torch.sigmoid,
    'softmax': lambda x: torch.nn.functional.softmax(x, dim=-1),
    'relu': lambda x: torch.nn.functional.relu(x, inplace=True),
    'none': lambda x: x,
})

# Configs
coco_base_config = Config({
    'dataset': coco_dataset,
    'num_classes': 81,
    'lr_steps': (280000, 360000, 400000),
    'max_iter': 400000,
    'max_num_detections': 100,
    
    # See mask_type for details.
    'mask_type': mask_type.direct,
    'mask_size': 16,
    'masks_to_train': 100,
    'mask_proto_src': None,
    'mask_proto_net': [(256, 3, {}), (256, 3, {})],
    'mask_proto_bias': False,
    'mask_proto_prototype_activation': activation_func.relu,
    'mask_proto_mask_activation': activation_func.sigmoid,
    'mask_proto_coeff_activation': activation_func.tanh,
    'mask_proto_crop': True,
    'mask_proto_loss': None,

    # This is filled in at runtime by Yolact's __init__, so don't touch it
    'mask_dim': None,

    # Input image size. If preserve_aspect_ratio is False, min_size is ignored.
    'min_size': 200,
    'max_size': 300,
    
    # Whether or not to do post processing on the cpu at test time (usually faster)
    'force_cpu_detect': True,
    'force_cpu_nms': True,

    # Whether or not to tie the mask loss to 0
    'train_masks': True,
    # If enabled, the gt masks will be cropped using the gt bboxes instead of the predicted ones.
    # This speeds up training time considerably but results in much worse mAP at test time.
    'use_gt_bboxes': False,

    # Whether or not to preserve aspect ratio when resizing the image.
    # If True, uses the faster r-cnn resizing scheme.
    # If False, all images are resized to max_size x max_size
    'preserve_aspect_ratio': False,

    # Whether or not to use the prediction module (c) from DSSD
    'use_prediction_module': False,

    # Whether or not to use the predicted coordinate scheme from Yolo v2
    'use_yolo_regressors': False,
    
    # For training, bboxes are considered "positive" if their anchors have a 0.5 IoU overlap
    # or greater with a ground truth box. If this is true, instead of using the anchor boxes
    # for this IoU computation, the matching function will use the predicted bbox coordinates.
    # Don't turn this on if you're not using yolo regressors!
    'use_prediction_matching': False,
    # The number of iterations to wait before starting prediction matching.
    'prediction_matching_delay': 100,

    'backbone': None,
    'name': 'base_config',
})

# Pretty close to the original ssd300 just using resnet101 instead of vgg16
ssd550_resnet101_config = coco_base_config.copy({
    'name': 'ssd550_resnet101',
    'backbone': resnet101_backbone.copy({
        'selected_layers': list(range(2, 8)),
        'pred_scales': [[5, 4]]*6,
        'pred_aspect_ratios': [ [[1], [1, sqrt(2), 1/sqrt(2), sqrt(3), 1/sqrt(3)][:n]] for n in [3, 5, 5, 5, 3, 3] ],
    }),

    'max_size': 550,
    'mask_size': 20, # Turned out not to help much

    'train_masks': True,
    'preserve_aspect_ratio': False,
    'use_prediction_module': False,
    'use_yolo_regressors': False,
})

ssd550_resnet101_yolo_matching_config = ssd550_resnet101_config.copy({
    'name': 'ssd550_resnet101_yolo_matching',

    'mask_size': 16,

    'use_yolo_regressors': True,
    'use_prediction_matching': True,

    # Because of prediction matching, the number of positives goes up to high and thus
    # we run out of memory when training masks. The amount of memory for training masks
    # is proportional to the number of positives after all.
    'train_masks': False,
})


# Close to vanilla ssd300
ssd300_config = coco_base_config.copy({
    'name': 'ssd300',
    'backbone': vgg16_backbone.copy({
        'selected_layers': [3] + list(range(5, 10)),
        'pred_scales': [[5, 4]]*6,
        'pred_aspect_ratios': [ [[1], [1, sqrt(2), 1/sqrt(2), sqrt(3), 1/sqrt(3)][:n]] for n in [3, 5, 5, 5, 3, 3] ],
    }),

    'max_size': 300,
    'mask_size': 20, # Turned out not to help much

    'train_masks': True,
    'preserve_aspect_ratio': False,
    'use_prediction_module': False,
    'use_yolo_regressors': False,
})

# Close to vanilla ssd300 but bigger!
ssd550_config = ssd300_config.copy({
    'name': 'ssd550',
    'backbone': ssd300_config.backbone.copy({
        'args': (vgg16_arch, [(256, 2), (256, 2), (128, 2), (128, 1), (128, 1)], [4]),
        'selected_layers': [4] + list(range(6, 11)),
    }),

    'max_size': 550,
    'mask_size': 16,
})

yolact_resnet101_config = ssd550_resnet101_config.copy({
    'name': 'yolact_resnet101',

    'train_masks': True,
    'preserve_aspect_ratio': False,
    'use_prediction_module': False,
    'use_yolo_regressors': True,
    'use_prediction_matching': False,

    'mask_type': mask_type.lincomb,
    'masks_to_train': 100,
    'mask_proto_src': 0,
    'mask_proto_net': [],
})

yolact_resnet101_dedicated_config = yolact_resnet101_config.copy({
    'name': 'yolact_resnet101_dedicated',
    'mask_proto_src': None,
    'mask_proto_net': [(256, 3, {'stride': 2})],
})

yolact_resnet101_deep_config = yolact_resnet101_config.copy({
    'name': 'yolact_resnet101_deep',
    'mask_proto_src': 0,
    'mask_proto_net': [(256, 3, {'stride': 2}), (256, 3, {'stride': 2})] + [(256, 3, {})] * 3,
})

yolact_resnet101_shallow_config = yolact_resnet101_config.copy({
    'name': 'yolact_resnet101_shallow',
    'mask_proto_src': 0,
    'mask_proto_net': [(256, 3, {'stride': 2}), (256, 3, {'stride': 2})],
})

yolact_resnet101_conv4_config = yolact_resnet101_config.copy({
    'name': 'yolact_resnet101_conv4',
    'mask_proto_src': 2,
    'mask_proto_net': [(256, 3, {'padding': 1})] * 5,
})

yolact_resnet101_deconv4_config = yolact_resnet101_config.copy({
    'name': 'yolact_resnet101_deconv4',
    'mask_proto_src': 2,
    'mask_proto_net': [(256, 3, {'padding': 1})] * 3 + [(256, -2, {'stride': 2})] * 2 + [(256, 3, {'padding': 1})],
})

yolact_resnet101_maskrcnn_config = yolact_resnet101_config.copy({
    'name': 'yolact_resnet101_maskrcnn',
    'mask_proto_src': 2,
    'mask_proto_net': [(256, 3, {'padding': 1})] * 4 + [(256, -2, {'stride': 2}), (256, 1, {})],
})

# Start of Ablations
yolact_resnet101_maskrcnn_1_config = yolact_resnet101_maskrcnn_config.copy({
    'name': 'yolact_resnet101_maskrcnn_1',
    'use_yolo_regressors': False,
})
yolact_resnet101_maskrcnn_2_config = yolact_resnet101_maskrcnn_config.copy({
    'name': 'yolact_resnet101_maskrcnn_2',
    'use_yolo_regressors': False,
    'mask_proto_prototype_activation': activation_func.sigmoid,
})
yolact_resnet101_maskrcnn_3_config = yolact_resnet101_maskrcnn_config.copy({
    'name': 'yolact_resnet101_maskrcnn_3',
    'use_yolo_regressors': False,
    'mask_proto_prototype_activation': activation_func.sigmoid,
    'use_prediction_module': True,
})
yolact_resnet101_maskrcnn_4_config = yolact_resnet101_maskrcnn_config.copy({
    'name': 'yolact_resnet101_maskrcnn_4',
    'use_yolo_regressors': False,
    'mask_proto_prototype_activation': activation_func.sigmoid,
    'use_prediction_module': True,
    'mask_proto_bias': True,
})
yolact_resnet101_maskrcnn_5_config = yolact_resnet101_maskrcnn_config.copy({
    'name': 'yolact_resnet101_maskrcnn_5',
    'use_yolo_regressors': False,
    'mask_proto_prototype_activation': activation_func.sigmoid,
    'use_prediction_module': True,
    'mask_proto_bias': True,
    'mask_proto_mask_activation': activation_func.none,
})
yolact_resnet101_maskrcnn_6_config = yolact_resnet101_maskrcnn_config.copy({
    'name': 'yolact_resnet101_maskrcnn_6',
    'use_yolo_regressors': False,
    'mask_proto_prototype_activation': activation_func.sigmoid,
    'use_prediction_module': True,
    'mask_proto_bias': True,
    'mask_proto_mask_activation': activation_func.relu,
})

# Ablations 2: Electric Boogaloo
yrm7_config = yolact_resnet101_maskrcnn_config.copy({
    'name': 'yrm7',
    'use_yolo_regressors': False,
    'mask_proto_coeff_activation': activation_func.sigmoid,
    'mask_proto_prototype_activation': activation_func.sigmoid,
    'mask_proto_mask_activation': activation_func.none,
})
yrm8_config = yolact_resnet101_maskrcnn_config.copy({
    'name': 'yrm8',
    'use_yolo_regressors': False,
    'mask_proto_coeff_activation': activation_func.softmax,
    'mask_proto_prototype_activation': activation_func.sigmoid,
    'mask_proto_mask_activation': activation_func.none,
})
yrm9_config = yolact_resnet101_maskrcnn_config.copy({
    'name': 'yrm9',
    'use_yolo_regressors': False,
    'mask_proto_coeff_activation': activation_func.sigmoid,
    'mask_proto_prototype_activation': activation_func.sigmoid,
    'mask_proto_mask_activation': activation_func.none,
    'mask_proto_crop': False,
})
yrm10_config = yolact_resnet101_maskrcnn_config.copy({
    'name': 'yrm10',
    'use_yolo_regressors': False,
    'mask_proto_loss': 'l1',
})
yrm11_config = yolact_resnet101_maskrcnn_config.copy({
    'name': 'yrm11',
    'use_yolo_regressors': False,
    'mask_proto_loss': 'disj',
})
yrm12_config = yolact_resnet101_maskrcnn_config.copy({
    'name': 'yrm12',
    'use_yolo_regressors': False,
    'mask_proto_coeff_activation': activation_func.none,
    'mask_proto_prototype_activation': activation_func.relu,
    'mask_proto_mask_activation': activation_func.sigmoid,
})
yrm13_config = yolact_resnet101_maskrcnn_config.copy({
    'name': 'yrm13',
    'use_yolo_regressors': False,
    'mask_proto_crop': False,
})

yolact_vgg16_config = ssd550_config.copy({
    'name': 'yolact_vgg16',

    'train_masks': True,
    'preserve_aspect_ratio': False,
    'use_prediction_module': False,
    'use_yolo_regressors': True,
    'use_prediction_matching': False,

    'mask_type': mask_type.lincomb,
    'masks_to_train': 100,
    'mask_proto_layer': 0,
})

cfg = yolact_resnet101_conv4_config.copy()

def set_cfg(config_name:str):
    """ Sets the active config. Works even if cfg is already imported! """
    global cfg

    # Note this is not just an eval because I'm lazy, but also because it can
    # be used like ssd300_config.copy({'max_size': 400}) for extreme fine-tuning
    cfg.replace(eval(config_name))

    
