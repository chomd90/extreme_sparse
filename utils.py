import argparse
import os
import sys
from datasets import get_dataset, DATASETS, get_num_classes
from time import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import SGD, Optimizer
from torch.optim.lr_scheduler import StepLR
import datetime
import time
import numpy as np
import copy
import types
from architectures import get_architecture
from math import ceil
from train_utils import AverageMeter, accuracy, accuracy_list

def train(loader: DataLoader, model: torch.nn.Module, criterion, optimizer: Optimizer, 
          epoch: int, device, print_freq=100, display=True):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()
#     print("Entered training function")

    # switch to train mode
    model.train()
    
    for i, (inputs, targets) in enumerate(loader):
        # measure data loading time
        data_time.update(time.time() - end)

        inputs = inputs.to(device)
        targets = targets.to(device)

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(acc1.item(), inputs.size(0))
        top5.update(acc5.item(), inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0 and display == True:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))

    return (losses.avg, top1.avg, top5.avg)

def test(loader: DataLoader, model: torch.nn.Module, criterion, device, print_freq, display=False):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    # switch to eval mode
    model.eval()

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(loader):
            # measure data loading time
            data_time.update(time.time() - end)

            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # compute output
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(acc1.item(), inputs.size(0))
            top5.update(acc5.item(), inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0 and display == True:
                print('Test : [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, top1=top1, top5=top5))

        return (losses.avg, top1.avg, top5.avg)

def model_inference(base_classifier, loader, device, display=False, print_freq=100):
    print_freq = 100
    top1 = AverageMeter()
    top5 = AverageMeter()

    start = time.time()
    base_classifier.eval()
    # Regular dataset:
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = base_classifier(inputs)
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            top1.update(acc1.item(), inputs.size(0))
            top5.update(acc5.item(), inputs.size(0))
            
            if i % print_freq == 0 and display == True:
                print("Test : [{0}/{1}]\t"
                      "Acc@1 {top1.avg:.3f}"
                      "Acc@5 {top5.avg:.3f}".format(
                      i, len(loader), top1=top1, top5=top5))
    end = time.time()
    if display == True:
        print("Inference Time: {0:.3f}".format(end-start))
        print("Final Accuracy: [{0}]".format(top1.avg))
        
    return top1.avg

def model_inference_imagenet(base_classifier, loader, device, display=False, print_freq=1000):
    print_freq = 100
    top1 = AverageMeter()
    top5 = AverageMeter()

    start = time.time()
    base_classifier.eval()
    # Regular dataset:
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(loader):
            inputs = inputs.to(device, non_blocking=True)
            targets = torch.tensor(targets)
            targets = targets.to(device, non_blocking=True)
            outputs = base_classifier(inputs)
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            top1.update(acc1.item(), inputs.size(0))
            top5.update(acc5.item(), inputs.size(0))
            
            if i % print_freq == 0 and display == True:
                print("Test : [{0}/{1}]\t"
                      "Acc@1 {top1.avg:.3f}"
                      "Acc@5 {top5.avg:.3f}".format(
                      i, len(loader), top1=top1, top5=top5))
    end = time.time()
    if display == True:
        print("Inference Time: {0:.3f}".format(end-start))
        print("Final Accuracy: [{0}]".format(top1.avg))
        
    return top1.avg, top5.avg

def mask_forward_conv2d(self, x):
    return F.conv2d(x, self.weight * self.weight_mask, self.bias, self.stride, 
                    self.padding, self.dilation, self.groups)

def mask_forward_linear(self, x):
    return F.linear(x, self.weight * self.weight_mask, self.bias)

def reset_forward_conv2d(self, x):
    return F.conv2d(x, self.weight, self.bias, self.stride, 
                    self.padding, self.dilation, self.groups)

def reset_forward_linear(self, x):
    return F.linear(x, self.weight, self.bias)


def apply_prune_mask(net, keep_masks):

    # Before I can zip() layers and pruning masks I need to make sure they match
    # one-to-one by removing all the irrelevant modules:
    prunable_layers = filter(
        lambda layer: isinstance(layer, nn.Conv2d) or isinstance(
            layer, nn.Linear), net.modules())

    for layer, keep_mask in zip(prunable_layers, keep_masks):
        assert (layer.weight.shape == keep_mask.shape)

        def hook_factory(keep_mask):
            """
            The hook function can't be defined directly here because of Python's
            late binding which would result in all hooks getting the very last
            mask! Getting it through another function forces early binding.
            """

            def hook(grads):
                return grads * keep_mask

            return hook

        # mask[i] == 0 --> Prune parameter
        # mask[i] == 1 --> Keep parameter

        # Step 1: Set the masked weights to zero (NB the biases are ignored)
        # Step 2: Make sure their gradients remain zero
        layer.weight.data[keep_mask == 0.] = 0.
        layer.weight.register_hook(hook_factory(keep_mask))
        
def mask_train(loader: DataLoader, model: torch.nn.Module, criterion, optimizer: Optimizer, 
               epoch: int, device, alpha, display=False):
    losses = AverageMeter()

    # switch to train mode
    model.train()
    
    for i, (inputs, targets) in enumerate(loader):

        inputs = inputs.to(device)
        targets = targets.to(device)
        
        reg_loss = 0
        for layer in model.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                reg_loss += torch.norm(layer.weight_mask, p=1)
        
        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets) + alpha * reg_loss

        losses.update(loss.item(), inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return losses.avg