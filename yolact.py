import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import Bottleneck
import numpy as np
from itertools import product
from math import sqrt

from data.config import cfg, mask_type
from layers import Detect
from layers.interpolate import InterpolateModule
from backbone import construct_backbone

import torch.backends.cudnn as cudnn
from utils import timer
from utils.functions import MovingAverage

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
        - aspect_ratios: A list of lists of priorbox aspect ratios (one list per scale).
        - scales:        A list of priorbox scales relative to this layer's convsize.
                         For instance: If this layer has convouts of size 30x30 for
                                       an image of size 600x600, the 'default' (scale
                                       of 1) for this layer would produce bounding
                                       boxes with an area of 20x20px. If the scale is
                                       .5 on the other hand, this layer would consider
                                       bounding boxes with area 10x10px, etc.
    """
    
    def __init__(self, in_channels, out_channels=1024, aspect_ratios=[[1]], scales=[1]):
        super().__init__()

        self.num_classes = cfg.num_classes
        self.mask_dim    = cfg.mask_dim
        self.num_priors  = sum(len(x) for x in aspect_ratios)
        
        if cfg.mask_proto_prototypes_as_features:
            out_channels += self.mask_dim

        if cfg.use_prediction_module:
            self.block = Bottleneck(in_channels, out_channels // 4)
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)
            self.bn = nn.BatchNorm2d(out_channels)

        self.bbox_layer = nn.Conv2d(out_channels, self.num_priors * 4,                 kernel_size=3, padding=1)
        self.conf_layer = nn.Conv2d(out_channels, self.num_priors * self.num_classes,  kernel_size=3, padding=1)
        self.mask_layer = nn.Conv2d(out_channels, self.num_priors * self.mask_dim,     kernel_size=3, padding=1)
        
        # What is this ugly lambda doing in the middle of all this clean prediction module code?
        def make_extra(num_layers):
            if num_layers == 0:
                return lambda x: x
            else:
                # Looks more complicated than it is. This just creates an array of num_layers alternating conv-relu
                return nn.Sequential(*sum([[
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True)
                ] for _ in range(num_layers)], []))

        self.bbox_extra, self.conf_extra, self.mask_extra = [make_extra(x) for x in cfg.extra_layers]
        
        if cfg.mask_type == mask_type.lincomb and cfg.mask_proto_coeff_gate:
            self.gate_layer = nn.Conv2d(out_channels, self.num_priors * self.mask_dim, kernel_size=3, padding=1)

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
            - mask_output: [batch_size, conv_h*conv_w*num_priors, mask_dim]
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

        bbox_x = self.bbox_extra(x)
        conf_x = self.conf_extra(x)
        mask_x = self.mask_extra(x)

        bbox = self.bbox_layer(bbox_x).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, 4)
        conf = self.conf_layer(conf_x).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, self.num_classes)
        mask = self.mask_layer(mask_x).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, self.mask_dim)
        
        # See box_utils.decode for an explaination of this
        if cfg.use_yolo_regressors:
            bbox[:, :, :2] = torch.sigmoid(bbox[:, :, :2]) - 0.5
            bbox[:, :, 0] /= conv_w
            bbox[:, :, 1] /= conv_h

        if cfg.mask_type == mask_type.direct:
            mask = torch.sigmoid(mask)
        elif cfg.mask_type == mask_type.lincomb:
            mask = cfg.mask_proto_coeff_activation(mask)

            if cfg.mask_proto_coeff_gate:
                gate = self.gate_layer(x).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, self.mask_dim)
                mask = mask * torch.sigmoid(gate)
        
        priors = self.make_priors(conv_h, conv_w)

        return (bbox, conf, mask, priors)
    
    def make_priors(self, conv_h, conv_w):
        """ Note that priors are [x,y,width,height] where (x,y) is the center of the box. """
        
        with timer.env('makepriors'):
            if self.last_conv_size != (conv_w, conv_h):
                prior_data = []

                # Iteration order is important (it has to sync up with the convout)
                for j, i in product(range(conv_h), range(conv_w)):
                    # +0.5 because priors are in center-size notation
                    x = (i + 0.5) / conv_w
                    y = (j + 0.5) / conv_h
                    
                    for scale, ars in zip(self.scales, self.aspect_ratios):
                        for ar in ars:
                            w = scale * ar / conv_w
                            h = scale / ar / conv_h

                            prior_data += [x, y, w, h]
                
                self.priors = torch.Tensor(prior_data).view(-1, 4)
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

        # Compute mask_dim here and add it back to the config. Make sure Yolact's constructor is called early!
        if cfg.mask_type == mask_type.direct:
            cfg.mask_dim = cfg.mask_size**2
        elif cfg.mask_type == mask_type.lincomb:
            if cfg.mask_proto_use_grid:
                self.grid = torch.Tensor(np.load(cfg.mask_proto_grid_file))
                self.num_grids = self.grid.size(0)
            else:
                self.num_grids = 0

            self.proto_src = cfg.mask_proto_src
            in_channels = 3 if self.proto_src is None else self.backbone.channels[self.proto_src]
            in_channels += self.num_grids

            def make_layer(layer_cfg):
                nonlocal in_channels
                num_channels = layer_cfg[0]
                kernel_size = layer_cfg[1]
                
                # Possible patterns:
                # ( 256, 3, {}) -> conv
                # ( 256,-2, {}) -> deconv
                # (None,-2, {}) -> bilinear interpolate
                #
                # You know it would have probably been simpler just to adopt a 'c' 'd' 'u' naming scheme.
                # Whatever, it's too late now.
                if kernel_size > 0:
                    layer = nn.Conv2d(in_channels, num_channels, kernel_size, **layer_cfg[2])
                else:
                    if num_channels is None:
                        layer = InterpolateModule(scale_factor=-kernel_size, mode='bilinear', align_corners=False, **layer_cfg[2])
                    else:
                        layer = nn.ConvTranspose2d(in_channels, num_channels, -kernel_size, **layer_cfg[2])
                
                in_channels = num_channels if num_channels is not None else in_channels
                return [layer, nn.ReLU(inplace=True)]

            # The -1 here is to remove the last relu because we might want to change it to another function
            self.proto_net = nn.Sequential(*(sum([make_layer(x) for x in cfg.mask_proto_net], [])[:-1]))
            cfg.mask_dim = in_channels

            if cfg.mask_proto_bias:
                cfg.mask_dim += 1

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

        if cfg.mask_type == mask_type.lincomb:
            with timer.env('proto'):
                proto_x = x if self.proto_src is None else outs[self.proto_src]
                
                if self.num_grids > 0:
                    grids = self.grid.repeat(proto_x.size(0), 1, 1, 1)
                    proto_x = torch.cat([proto_x, grids], dim=1)

                proto_out = self.proto_net(proto_x)
                proto_out = cfg.mask_proto_prototype_activation(proto_out)
                
                if cfg.mask_proto_prototypes_as_features:
                    proto_downsampled = proto_out.clone()
                
                # Move the features last so the multipliaction is easy
                proto_out = proto_out.permute(0, 2, 3, 1).contiguous()

                if cfg.mask_proto_bias:
                    bias_shape = [x for x in proto_out.size()]
                    bias_shape[-1] = 1
                    proto_out = torch.cat([proto_out, torch.ones(*bias_shape)], -1)

        with timer.env('pass2'):
            pred_outs = ([], [], [], [])
            
            for idx, pred_layer in zip(self.selected_layers, self.prediction_layers):
                pred_x = outs[idx]

                if cfg.mask_type == mask_type.lincomb and cfg.mask_proto_prototypes_as_features:
                    # Scale the prototypes down to the current prediction layer's size and add it as inputs
                    proto_downsampled = F.interpolate(proto_downsampled, size=outs[idx].size()[2:], mode='bilinear', align_corners=False)
                    pred_x = torch.cat([pred_x, proto_downsampled], dim=1)

                p = pred_layer(pred_x)
                
                for out, pred in zip(pred_outs, p):
                    out.append(pred)

        pred_outs = [torch.cat(x, -2) for x in pred_outs]

        if cfg.mask_type == mask_type.lincomb:
            pred_outs.append(proto_out)

        if self.training:
            return pred_outs
        else:
            pred_outs[1] = F.softmax(pred_outs[1], -1) # Softmax the conf output
            return self.detect(*pred_outs)




# Some testing code
if __name__ == '__main__':
    from utils.functions import init_console
    init_console()

    net = Yolact()
    net.train()
    net.init_weights(backbone_path='weights/' + cfg.backbone.path)

    # GPU
    net = net.cuda()
    cudnn.benchmark = True
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    x = torch.zeros((1, 3, cfg.max_size, cfg.max_size))
    y = net(x)

    for p in net.prediction_layers:
        print(p.last_conv_size)

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
            print('\033[2J') # Moves console cursor to 0,0
            timer.print_stats()
            print('Avg fps: %.2f\tAvg ms: %.2f         ' % (1/avg.get_avg(), avg.get_avg()*1000))
    except KeyboardInterrupt:
        pass
