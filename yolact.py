import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import Bottleneck
import numpy as np

from data.config import get_cfg
from layers import Detect
from backbone import construct_backbone

import torch.backends.cudnn as cudnn
from utils import timer
from utils.functions import MovingAverage

cfg = get_cfg()

class PredictionModule(nn.Module):
    """
    The (c) prediction module adapted from DSSD:
    https://arxiv.org/pdf/1701.06659.pdf

    Note that this is slightly different to the module in the paper
    because the Bottleneck block actually has a 3x3 convolution in
    the middle instead of a 1x1 convolution. Though, I really can't
    be arsed to implement it myself, and, who knows, this might be
    better.

    Args:
        - in_channels:   The input feature size.
        - out_channels:  The output feature size (must be a multiple of 4).
        - aspect_ratios: A list of priorbox aspect ratios to consider.
        - scales:        A list of priorbox scales relative to this layer's convsize.
                         For instance: If this layer has convouts of size 30x30 for
                                       an image of size 600x600, the 'default' (scale
                                       of 1) for this layer would produce bounding
                                       boxes with an area of 20x20px. If the scale is
                                       .5 on the other hand, this layer would consider
                                       bounding boxes with area 10x10px, etc.
        - num_classes:   The number of classes to consider for classification.
        - mask_size:     The side length of the downsampled predicted mask.
    """
    
    def __init__(self, in_channels, out_channels=1024, aspect_ratios=[1], scales=[1],
                       num_classes=cfg.num_classes, mask_size=cfg.mask_size):
        super().__init__()

        self.num_classes = num_classes
        self.mask_size   = mask_size
        self.num_priors  = len(aspect_ratios) * len(scales)

        if cfg.use_prediction_module:
            self.block = Bottleneck(in_channels, out_channels // 4)
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)
            self.bn = nn.BatchNorm2d(out_channels)

        self.bbox_layer = nn.Conv2d(out_channels, self.num_priors * 4,              kernel_size=3, padding=1)
        self.conf_layer = nn.Conv2d(out_channels, self.num_priors * num_classes,    kernel_size=3, padding=1)
        self.mask_layer = nn.Conv2d(out_channels, self.num_priors * (mask_size**2), kernel_size=3, padding=1)

        self.aspect_ratios = aspect_ratios
        self.scales = scales

        self.priors = None
        self.last_conv_size = None

    def forward(self, x):
        """
        Args:
            - x: The convOut from a layer in the backbone network
                 Size: [batch_size, in_channels, conv_h, conv_w])

        Returns a tuple (bbox_coords, class_confs, mask_output, prior_boxes) with sizes
            - bbox_coords: [batch_size, conv_h*conv_w*num_priors, 4]
            - class_confs: [batch_size, conv_h*conv_w*num_priors, num_classes]
            - mask_output: [batch_size, conv_h*conv_w*num_priors, mask_size**2]
            - prior_boxes: [conv_h*conv_w*num_priors, 4]
        """
        conv_h = x.size(2)
        conv_w = x.size(3)
        
        if cfg.use_prediction_module:
            # The two branches of PM design (c)
            a = self.block(x)
            
            b = self.conv(x)
            b = self.bn(b)
            b = F.relu(b)
            
            # TODO: Possibly switch this out for a product
            x = a + b

        bbox = self.bbox_layer(x).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, 4)
        conf = self.conf_layer(x).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, self.num_classes)
        mask = self.mask_layer(x).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, self.mask_size**2)
        
        # See box_utils.decode for an explaination of this
        if cfg.use_yolo_regressors:
            bbox[:, :, :2] = F.sigmoid(bbox[:, :, :2]) - 0.5
            bbox[:, :, 0] /= conv_w
            bbox[:, :, 1] /= conv_h


        mask = F.sigmoid(mask)
        
        priors = self.make_priors(conv_h, conv_w)

        return (bbox, conf, mask, priors)
    
    def make_priors(self, conv_h, conv_w):
        """ Note that priors are [x,y,width,height] where (x,y) is the center of the box. """
        with timer.env('makepriors'):
            if self.last_conv_size != (conv_w, conv_h):
                # Fancy fast way of doing a cartesian product
                priors = np.array(np.meshgrid(list(range(conv_w)),
                                              list(range(conv_h)),
                                              self.aspect_ratios,
                                              self.scales), dtype=np.float32).T.reshape(-1, 4)
                
                # The predictions will be in the order conv_h, conv_w, num_priors, but I don't
                # know if meshgrid ordering is deterministic, so let's sort it here to make sure
                # the elements are in this order. aspect_ratios and scales are interchangable
                # because the network can just learn which is which, but conv_h and conv_w orders
                # have to match or the network will get priors for a cell that is in the complete
                # wrong place in the image. Note: the sort order is from last to first (so 1, 0, etc.)
                ind = np.lexsort((priors[:,3],priors[:,2],priors[:,0],priors[:,1]), axis=0)
                priors = priors[ind]
                
                # Priors are in center-size form
                priors[:, [0, 1]] += 0.5

                # Compute the correct width and height of each bounding box
                aspect_ratios = priors[:, 2].copy() # In the form w / h
                scales        = priors[:, 3].copy()
                priors[:, 2] = scales * aspect_ratios # p_w = (scale / h) * w
                priors[:, 3] = scales / aspect_ratios # p_h = (scale / w) * h

                # Make those coordinate relative
                priors[:, [0, 2]] /= conv_w
                priors[:, [1, 3]] /= conv_h
                
                # Cache priors because copying them to the gpu takes time
                self.priors = torch.Tensor(priors)
                self.last_conv_size = (conv_w, conv_h)
        
        return self.priors



class Yolact(nn.Module):
    """


    ██╗   ██╗ ██████╗ ██╗      █████╗  ██████╗████████╗
    ╚██╗ ██╔╝██╔═══██╗██║     ██╔══██╗██╔════╝╚══██╔══╝
     ╚████╔╝ ██║   ██║██║     ███████║██║        ██║   
      ╚██╔╝  ██║   ██║██║     ██╔══██║██║        ██║   
       ██║   ╚██████╔╝███████╗██║  ██║╚██████╗   ██║   
       ╚═╝    ╚═════╝ ╚══════╝╚═╝  ╚═╝ ╚═════╝   ╚═╝ 


    You can set the arguments by chainging them in the backbone config object in config.py.

    Parameters (in cfg.backbone):
        - selected_layers: The indices of the conv layers to use for prediction.
        - pred_scales:     A list with len(selected_layers) containing tuples of scales (see PredictionModule)
        - pred_aspect_ratios: A list of lists of aspect ratios with len(selected_layers) (see PredictionModule)
    """

    def __init__(self):
        super().__init__()

        selected_layers    = cfg.backbone.selected_layers
        pred_scales        = cfg.backbone.pred_scales
        pred_aspect_ratios = cfg.backbone.pred_aspect_ratios

        self.backbone = construct_backbone(cfg.backbone)     

        self.selected_layers = selected_layers
        self.prediction_layers = nn.ModuleList()

        for idx, layer_idx in enumerate(self.selected_layers):
            pred = PredictionModule(self.backbone.channels[layer_idx], self.backbone.channels[layer_idx],
                                    aspect_ratios=pred_aspect_ratios[idx], scales=pred_scales[idx])
            self.prediction_layers.append(pred)

        # For use in evaluation
        self.detect = Detect(cfg.num_classes, bkg_label=0, top_k=200, conf_thresh=0.01, nms_thresh=0.45)

    def save_weights(self, path):
        """ Saves the model's weights using compression because the file sizes were getting too big. """
        torch.save(self.state_dict(), path)
    
    def load_weights(self, path):
        """ Loads weights from a compressed save file. """
        self.load_state_dict(torch.load(path))

    def init_weights(self, backbone_path):
        """ Initialize weights for training. """
        # Initialize the backbone with the pretrained weights.
        self.backbone.init_backbone(backbone_path)

        # Initialize the rest of the conv layers with xavier
        for module in self.modules():
            if isinstance(module, nn.Conv2d) and module not in self.backbone.backbone_modules:
                nn.init.xavier_uniform_(module.weight.data)
                if module.bias is not None:
                    module.bias.data.zero_()

    def forward(self, x):
        """ The input should be of size [batch_size, 3, img_h, img_w] """
        with timer.env('pass1'):
            outs = self.backbone(x)

        with timer.env('pass2'):
            pred_outs = ([], [], [], [])
            for idx, pred_layer in zip(self.selected_layers, self.prediction_layers):
                p = pred_layer(outs[idx])
                for out, pred in zip(pred_outs, p):
                    out.append(pred)

        pred_outs = [torch.cat(x, -2) for x in pred_outs]

        if self.training:
            return pred_outs
        else:
            pred_outs[1] = F.softmax(pred_outs[1], -1) # Softmax the conf output
            return self.detect(*pred_outs)


if __name__ == '__main__':
    from utils.functions import init_console
    init_console()

    net = Yolact()
    net.train()
    net.init_weights(backbone_path='weights/' + cfg.backbone.path)

    # GPU
    # net = net.cuda()
    # cudnn.benchmark = True
    # torch.set_default_tensor_type('torch.cuda.FloatTensor')

    x = torch.zeros((1, 3, cfg.max_size, cfg.max_size))

    y = net(x)
    print()
    for a in y:
        print(a.size(), torch.sum(a))
    exit()
    
    net(x)
    # timer.disable('pass2')
    avg = MovingAverage()
    try:
        while True:
            timer.reset()
            with timer.env('everything else'):
                net(x)
            avg.add(timer.total_time())
            print('\033[2J')
            timer.print_stats()
            print('Avg fps: %.2f\tAvg ms: %.2f         ' % (1/avg.get_avg(), avg.get_avg()*1000))
    except KeyboardInterrupt:
        pass
