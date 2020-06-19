import argparse
import os
from datasets import get_dataset, DATASETS, get_num_classes
from time import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import SGD, Optimizer
from torch.optim.lr_scheduler import StepLR, MultiStepLR
import datetime
import time
import numpy as np
import copy
import types
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from architectures import get_architecture, ARCHITECTURES
from math import ceil
from train_utils import AverageMeter, accuracy, accuracy_list, init_logfile, log
from utils import *
import utils

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
from ignite.contrib.handlers import ProgressBar

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data_train', metavar='DIR',
                    help='path to train dataset')
parser.add_argument('data_val', metavar='DIR', help='path to valid dataset')
parser.add_argument('arch', type=str, default="resnet50")
parser.add_argument('outdir', type=str, help='folder to save model and training log)')
parser.add_argument('--keep_mask', type=str, default='rewind_10percent_keep_mask.pt')
parser.add_argument('--save_model', type=str, default='rewind_10percent_save_model.pt')
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=160, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--epochs_warmup', default=10, type=int)
parser.add_argument('--batch', default=256, type=int, metavar='N',
                    help='batchsize (default: 256)')
parser.add_argument('--logname', type=str, default='log.txt')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    help='initial learning rate', dest='lr')
parser.add_argument('--alpha', default=1e-4, type=float,
                    help='Lasso coefficient')
parser.add_argument('--keep_ratio', default=0.1, type=float)
parser.add_argument('--round', default=1, type=int)
parser.add_argument('--lr_step_size', type=int, default=30,
                    help='How often to decrease learning by gamma.')
parser.add_argument('--gamma', type=float, default=0.1,
                    help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--gpu', default=0, type=int,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--print-freq', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')
args = parser.parse_args()

def main():
    
    if not os.path.exists(args.outdir):
        os.mkdir(args.outdir)

    device = torch.device("cuda")
    torch.cuda.set_device(args.gpu)

    logfilename = os.path.join(args.outdir, args.logname)


    init_logfile(logfilename, "epoch\ttime\tlr\ttrain loss\ttrain acc\ttestloss\ttest acc")
    log(logfilename, "Hyperparameter List")
    log(logfilename, "Epochs: {:}".format(args.epochs))
    log(logfilename, "Learning Rate: {:}".format(args.lr))
    log(logfilename, "Alpha: {:}".format(args.alpha))
    log(logfilename, "Keep ratio: {:}".format(args.keep_ratio))
    log(logfilename, "Warmup Epochs: {:}".format(args.epochs_warmup))

    test_acc_list = []
    for _ in range(args.round):
        traindir = os.path.join(args.data_train, 'train')
        valdir = os.path.join(args.data_val, 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))

        train_sampler = None

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch, shuffle=(train_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=train_sampler)

        test_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=args.batch, shuffle=False,
            num_workers=args.workers, pin_memory=True)


        base_classifier = models.__dict__[args.arch](pretrained=False).cuda()
        print("Loaded the base_classifier")
    
        criterion = nn.CrossEntropyLoss().to(device)
        optimizer = SGD(base_classifier.parameters(), lr=args.lr, momentum=args.momentum,
                        weight_decay=args.weight_decay)
        
        # Warmup training for the rewinding.
        for epoch in range(args.epochs_warmup):
            print("Warmup Training Epochs: {:}".format(epoch))
            log(logfilename, "Warmup current epochs: {}".format(epoch))
            train_loss, train_top1, train_top5 = utils.train(train_loader, 
                                                       base_classifier,
                                                       criterion,
                                                       optimizer,
                                                       epoch,
                                                       device,
                                                       print_freq=100,
                                                       display=True)

        original_acc = model_inference(base_classifier, test_loader,
                                       device, display=True)
        log(logfilename, "Warmup Model Test Accuracy: {:.5}".format(original_acc))
        print("Warmup Model Test Accuracy, ", original_acc)

        # Creating a fresh copy of network not affecting the original network.
        # Goal is to find the supermask.
        
        net = copy.deepcopy(base_classifier)
        net = net.to(device)
        
        # Generating the mask 'm'
        for layer in net.modules():
            if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
                layer.weight_mask = nn.Parameter(torch.ones_like(layer.weight))

                layer.weight.requires_grad = True
                layer.weight_mask.requires_grad = True

            # This is the monkey-patch overriding layer.forward to custom function.
            # layer.forward will pass nn.Linear with weights: 'w' and 'm' elementwised
            if isinstance(layer, nn.Linear):
                layer.forward = types.MethodType(mask_forward_linear, layer)

            if isinstance(layer, nn.Conv2d):
                layer.forward = types.MethodType(mask_forward_conv2d, layer)


        criterion = nn.CrossEntropyLoss().to(device)    # Criterion for training the mask.
        optimizer = SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
                        weight_decay=0) 
          # weight_decay = 0 for training the mask.
#         warm_scheduler = StepLR(optimizer, step_size=args.epochs_mask-10, gamma=0.2)


        sparsity, total = 0, 0
        breakFlag = False
        net.train()
        # Training the mask with the training set.
        for epoch in range(100000):
#             if epoch % 5 == 0:
            print("Current epochs: ", epoch)
            print("Sparsity: {:}".format(sparsity))
            log(logfilename, "Current epochs: {}".format(epoch))
            log(logfilename, "Sparsity: {:}".format(sparsity))
                
            for i, (inputs, targets) in enumerate(train_loader):
                inputs = inputs.cuda()
                targets = targets.cuda()

                reg_loss = 0
                for layer in net.modules():
                    if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                        reg_loss += torch.norm(layer.weight_mask, p=1)
                outputs = net(inputs)
                loss = criterion(outputs, targets) + args.alpha * reg_loss
            
                # Computing gradient and do SGD
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
                sparsity, total = 0, 0
                for layer in net.modules():
                    if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
                        boolean_list = layer.weight_mask.data > 1e-3
                        sparsity += (boolean_list == 1).sum()
                        total += layer.weight.numel()
                        
                if i % 50 == 0:
                    print("Current Epochs: {}, Current i: {}, Current Sparsity: {}".format(epoch, i, sparsity))
                
                if sparsity <= total*args.keep_ratio:
                    print("Current epochs breaking loop at {:}".format(epoch))
                    log(logfilename, "Current epochs breaking loop at {:}".format(epoch))
                    breakFlag = True
                    break
#                 if breakFlag == True:
#                     break
            if breakFlag == True:
                break
            
#                     print("W 1-norm: ", torch.norm(layer.weight_mask, p=1))

        # Just checking the 1-norm of weights in each layer.
        # Approximates how sparse the mask is..
                
        # This line allows to calculate the threshold to satisfy the keep_ratio.
        c_abs = []
        for layer in net.modules():
            if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
                c_abs.append(torch.abs(layer.weight_mask))
        
        all_scores = torch.cat([torch.flatten(x) for x in c_abs])
        num_params_to_keep = int(len(all_scores) * args.keep_ratio)
        threshold, _ = torch.topk(all_scores, num_params_to_keep, sorted=True)
        threshold = threshold[-1]
        
        print("Threshold found: ", threshold)
        
        keep_masks = []
        for c in c_abs:
            keep_masks.append((c >= threshold).float())
        print("Number of ones.", torch.sum(torch.cat([torch.flatten(x == 1) for x in keep_masks])))
        
        
        torch.save(base_classifier.state_dict(), os.path.join(args.outdir, args.save_model))
        base_classifier_acc = model_inference(base_classifier, test_loader, device, display=True)
        log(logfilename, "Weight Update Test Accuracy: {:.5}".format(base_classifier_acc))
        print("Saved the rewind model.")
        for masks in keep_masks:
            masks = masks.data
            
        torch.save(keep_masks, os.path.join(args.outdir, args.keep_mask))
        print("Saved the masking function.")
        log(logfilename, "Finished finding the mask. (REWIND)")


if __name__ == "__main__":
    main()


