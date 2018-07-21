from data import *
from utils.augmentations import SSDAugmentation
from utils.functions import MovingAverage
from layers.modules import MultiBoxLoss
from yolact import Yolact
import os
import sys
import time
import math
from pathlib import Path
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
import numpy as np
import argparse
import datetime


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(
    description='Yolact Training Script')
parser.add_argument('--dataset_root', default=COCO_ROOT,
                    help='Dataset root directory path')
parser.add_argument('--batch_size', default=32, type=int,
                    help='Batch size for training')
parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--start_iter', default=0, type=int,
                    help='Resume training at this iter')
parser.add_argument('--num_workers', default=4, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--visdom', default=False, type=str2bool,
                    help='Use visdom for loss visualization')
parser.add_argument('--save_folder', default='weights/',
                    help='Directory for saving checkpoint models')
parser.add_argument('--model_name', default='yolact',
                    help='The name of the model used for saving checkpoints')
args = parser.parse_args()


if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

def train():
    if not os.path.exists(args.save_folder):
        os.mkdir(args.save_folder)

    cfg = get_cfg()
    dataset = COCODetection(root=args.dataset_root,
                            image_set=cfg.dataset.split,
                            transform=SSDAugmentation(MEANS))

    if args.visdom:
        import visdom
        global viz
        viz = visdom.Visdom(port=8091)

    # Parallel wraps the underlying module, but when saving and loading we don't want that
    yolact_net = Yolact()
    net = yolact_net

    if args.resume:
        print('Resuming training, loading {}...'.format(args.resume))
        yolact_net.load_weights(args.resume)
    else:
        print('Loading base network...')
        yolact_net.load_state_dict(torch.load(args.save_folder + cfg.backbone.path), strict=False)

    if args.cuda:
        cudnn.benchmark = True
        net = torch.nn.DataParallel(net).cuda()

    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay)
    criterion = MultiBoxLoss(cfg.num_classes, 0.5, True, 0, True, 3, 0.5, False, args.cuda)

    net.train()

    # loss counters
    loc_loss = 0
    conf_loss = 0
    iteration = args.start_iter

    epoch_size = len(dataset) // args.batch_size
    num_epochs = math.ceil(cfg.max_iter / epoch_size)
    
    step_index = 0

    if args.visdom:
        vis_title = 'Training yolact with config %s' % cfg.name
        vis_legend = ['Loc Loss', 'Conf Loss', 'Total Loss']
        iter_plot = create_vis_plot('Iteration', 'Loss', vis_title, vis_legend)
        epoch_plot = create_vis_plot('Epoch', 'Loss', vis_title, vis_legend)

    data_loader = data.DataLoader(dataset, args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=True, collate_fn=detection_collate,
                                  pin_memory=True)
    
    
    save_path = lambda epoch, iteration: args.save_folder + cfg.name + '_' + str(epoch) + '_' + str(iteration) + '.pth'
    time_avg = MovingAverage()

    print('Begin training!')
    print()
    # try-except so you can use ctrl+c to save early and stop training
    try:
        for epoch in range(num_epochs):
            # Resume from start_iter
            if (epoch+1)*epoch_size < iteration:
                continue

            
            for datum in data_loader:
                # Stop if we've reached an epoch if we're resuming from start_iter
                if iteration == (epoch+1)*epoch_size:
                    break

                # Stop at the configured number of iterations even if mid-epoch
                if iteration == cfg.max_iter:
                    break

                if iteration in cfg.lr_steps:
                    step_index += 1
                    adjust_learning_rate(optimizer, args.gamma, step_index)

                # load train data
                images, targets_tuple = datum
                targets, masks = targets_tuple

                if args.cuda:
                    images = Variable(images.cuda())
                    targets = [Variable(ann.cuda()) for ann in targets]
                    masks = [Variable(mask.cuda()) for mask in masks]
                else:
                    images = Variable(images)
                    targets = [Variable(ann) for ann in targets]
                    masks = [Variable(mask) for mask in masks]
                # forward
                t0 = time.time()
                out = net(images)
                # backprop
                optimizer.zero_grad()
                loss_l, loss_c, loss_m = criterion(out, targets, masks)
                loss = loss_l + loss_c + loss_m
                loss.backward()
                optimizer.step()
                t1 = time.time()
                loc_loss += loss_l.item()
                conf_loss += loss_c.item()

                if iteration != args.start_iter:
                    time_avg.add(t1 - t0)

                if iteration % 10 == 0:
                    print('timer: %.4f sec.' % (t1 - t0))
                    eta_str = datetime.timedelta(seconds=(cfg.max_iter-iteration) * time_avg.get_avg())
                    print('epoch ' + repr(epoch) + ' || iter ' + repr(iteration) + ' || Loss: %.4f || Mask Loss: %.4f || ETA: %s ||'
                            % (loss.item(), loss_m.item(), eta_str), end=' ')

                if args.visdom:
                    update_vis_plot(iteration, loss_l.item(), loss_c.item(),
                                    iter_plot, epoch_plot, 'append')
                
                iteration += 1

                if iteration % 10000 == 0 and iteration != args.start_iter:
                    print('Saving state, iter:', iteration)
                    yolact_net.save_weights(save_path(epoch, iteration))
                

            if args.visdom:
                    update_vis_plot(epoch, loc_loss, conf_loss, epoch_plot, None,
                                    'append', epoch_size)
                    # reset epoch loss counters
                    loc_loss = 0
                    conf_loss = 0
    except KeyboardInterrupt:
        print('Stopping early. Saving network...')
        
        # Delete previous copy of the interrupted network so we don't spam the weights folder
        for p in Path(args.save_folder).glob('*_interrupt.pth'):
            p.unlink()
        
        yolact_net.save_weights(save_path(epoch, repr(iteration) + '_interrupt'))
        exit()

    yolact_net.save_weights(save_path(epoch, iteration))


def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def xavier(param):
    init.xavier_uniform_(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()


def create_vis_plot(_xlabel, _ylabel, _title, _legend):
    return viz.line(
        X=torch.zeros((1,)).cpu(),
        Y=torch.zeros((1, 3)).cpu(),
        opts=dict(
            xlabel=_xlabel,
            ylabel=_ylabel,
            title=_title,
            legend=_legend
        )
    )


def update_vis_plot(iteration, loc, conf, window1, window2, update_type,
                    epoch_size=1):
    viz.line(
        X=torch.ones((1, 3)).cpu() * iteration,
        Y=torch.Tensor([loc, conf, loc + conf]).unsqueeze(0).cpu() / epoch_size,
        win=window1,
        update=update_type
    )
    # initialize epoch plot on first iteration
    if iteration == 0:
        viz.line(
            X=torch.zeros((1, 3)).cpu(),
            Y=torch.Tensor([loc, conf, loc + conf]).unsqueeze(0).cpu(),
            win=window2,
            update=True
        )


if __name__ == '__main__':
    train()
