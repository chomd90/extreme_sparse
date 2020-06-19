import argparse
import os
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
from architectures import get_architecture, ARCHITECTURES
from math import ceil
from train_utils import AverageMeter, accuracy, accuracy_list, init_logfile, log
from utils import *

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
from ignite.contrib.handlers import ProgressBar

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('dataset', type=str, choices=DATASETS)
parser.add_argument('arch', type=str, choices=ARCHITECTURES)
parser.add_argument('outdir', type=str, help='folder to save model and training log)')
parser.add_argument('savedir', type=str, help='folder to load model')
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=50, type=int,
                    help='number of total epochs for the finetuning')
parser.add_argument('--batch', default=256, type=int, metavar='N',
                    help='batchsize (default: 256)')
parser.add_argument('--logname', type=str, default='log.txt')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    help='initial learning rate', dest='lr')
parser.add_argument('--alpha', default=1e-4, type=float,
                    help='Lasso coefficient')
parser.add_argument('--threshold', default=1e-2, type=float)
parser.add_argument('--keep_ratio', default=0.01, type=float)
parser.add_argument('--round', default=1, type=int)
parser.add_argument('--lr_step_size', type=int, default=30,
                    help='How often to decrease learning by gamma.')
parser.add_argument('--gamma', type=float, default=0.1,
                    help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
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

    log(logfilename, "Hyperparameter List")
    log(logfilename, "Epochs: {:}".format(args.epochs))
    log(logfilename, "Learning Rate: {:}".format(args.lr))
    log(logfilename, "Alpha: {:}".format(args.alpha))
    log(logfilename, "Keep ratio: {:}".format(args.keep_ratio))

    test_acc_list = []
    for _ in range(args.round):
        train_dataset = get_dataset(args.dataset, 'train')
        test_dataset = get_dataset(args.dataset, 'test')
        pin_memory = (args.dataset == "imagenet")
        train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch,
                                  num_workers=args.workers, pin_memory=pin_memory)
        test_loader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch,
                                 num_workers=args.workers, pin_memory=pin_memory)


        # Loading the base_classifier
        base_classifier = get_architecture(args.arch, args.dataset, device)
        checkpoint = torch.load(args.savedir)
        base_classifier.load_state_dict(checkpoint['state_dict'])
        base_classifier.eval()
        print("Loaded the base_classifier")

        original_acc = model_inference(base_classifier, test_loader,
                                       device, display=True)
        
        log(logfilename, "Original Model Test Accuracy: {:.5}".format(original_acc))
        print("Original Model Test Accuracy, ", original_acc)

        # Creating a fresh copy of network not affecting the original network.
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


        criterion = nn.NLLLoss().to(device)    # I added Log Softmax layer to all architecture.
        optimizer = SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
                        weight_decay=0) # weight_decay = 0 for training the mask.
 
        sparsity, total = 0, 0
        for layer in net.modules():
            if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
                boolean_list = layer.weight_mask.data > args.threshold
                sparsity += (boolean_list == 1).sum()
                total += layer.weight.numel()
        
        # Training the mask with the training set.
        # You can set the maximum number of loop in case the sparsity on auxiliary parameter 
        # do not go below target sparsity.
        for epoch in range(300):
            if epoch % 5 == 0:
                print("Current epochs: ", epoch)
                print("Sparsity: {:}".format(sparsity))
            train_loss = mask_train(train_loader, net, criterion, optimizer,
                                    epoch, device, alpha=args.alpha, display=False)
            acc = model_inference(net, test_loader, device, display=False)
            log(logfilename, "Epoch {:}, Mask Update Test Acc: {:.5}".format(epoch, acc))

            sparsity, total = 0, 0
            for layer in net.modules():
                if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
                    boolean_list = layer.weight_mask.data > args.threshold
                    sparsity += (boolean_list == 1).sum()
                    total += layer.weight.numel()
                    
            if sparsity <= total*args.keep_ratio:
                print("Current epochs breaking loop at {:}".format(epoch))
                break
        
        mask_update_acc = model_inference(net, test_loader, device, display=True)
        log(logfilename, "Mask Update Test Accuracy: {:.5}".format(mask_update_acc))
                
        # This line allows to calculate the threshold to satisfy the keep_ratio.
        c_abs = []
        for layer in net.modules():
            if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
                c_abs.append(torch.abs(layer.weight_mask))
        
        all_scores = torch.cat([torch.flatten(x) for x in c_abs])
        num_params_to_keep = int(len(all_scores) * args.keep_ratio)
        threshold, _ = torch.topk(all_scores, num_params_to_keep, sorted=True)
        threshold = threshold[-1]
        
        keep_masks = []
        for c in c_abs:
            keep_masks.append((c >= threshold).float())
        print("Number of ones.", torch.sum(torch.cat([torch.flatten(x == 1) for x in keep_masks])))
        
        # Updating the weight with elementwise product of update c.
        for layer in net.modules():
            if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
                # We update the weight by elementwise multiplication between
                # weight 'w' and mask 'm'.
                layer.weight.data = layer.weight.data * layer.weight_mask.data
                layer.zeros = nn.Parameter(torch.zeros_like(layer.weight))    # Dummy parameter.
                layer.ones = nn.Parameter(torch.ones_like(layer.weight))      # Dummy parameter.
                layer.weight_mask.data = torch.where(torch.abs(layer.weight_mask) <= threshold,
                                                layer.zeros,
                                                layer.ones)    # Updated weight_mask becomes the mask with element
                                                               # 0 and 1 again.

                # Temporarily disabling the backprop for both 'w' and 'm'.
                layer.weight.requires_grad = False
                layer.weight_mask.requires_grad = False

            if isinstance(layer, nn.Linear):
                layer.forward = types.MethodType(mask_forward_linear, layer)

            if isinstance(layer, nn.Conv2d):
                layer.forward = types.MethodType(mask_forward_conv2d, layer)

        weight_update_acc = model_inference(net, test_loader, device, display=True)
        log(logfilename, "Weight Update Test Accuracy: {:.5}".format(weight_update_acc))
        
        
        # Calculating the sparsity of the network.
        remain = 0
        total = 0
        for layer in net.modules():
            if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
                total += torch.norm(torch.ones_like(layer.weight), p=1)    # Counting total num parameter
                remain += torch.norm(layer.weight_mask.data, p=1)          # Counting ones in the mask.

                # Disabling backprop except weight 'w' for the finetuning.
                layer.zeros.requires_grad = False
                layer.ones.requires_grad = False
                layer.weight_mask.requires_grad = False
                layer.weight.requires_grad = True

            if isinstance(layer, nn.Linear):
                layer.forward = types.MethodType(mask_forward_linear, layer)

            if isinstance(layer, nn.Conv2d):
                layer.forward = types.MethodType(mask_forward_conv2d, layer)

        log(logfilename, "Sparsity: {:.3}".format(remain/total))
        print("Sparsity: ", remain/total)

#        --------------------------------
        # We need to transfer the weight we learned from "net" to "base_classifier".
        for (layer1, layer2) in zip(base_classifier.modules(), net.modules()):
            if isinstance(layer1, (nn.Linear, nn.Conv2d)) or isinstance(layer2, (nn.Linear, nn.Conv2d)):
                layer1.weight.data = layer2.weight.data
                if layer1.bias != None:
                    layer1.bias.data = layer2.bias.data
                    layer1.bias.requires_grad = True

                layer1.weight.requires_grad = True
                
        # Applying the mask to the base_classifier.
        apply_prune_mask(base_classifier, keep_masks)
#        --------------------------------

        optimizer = SGD(base_classifier.parameters(), lr=1e-3,
                        momentum=args.momentum, weight_decay=args.weight_decay)
        loss = nn.NLLLoss()
        scheduler = StepLR(optimizer, step_size=args.lr_step_size, gamma=args.gamma)
        
        
        test_acc = []
        # Finetuning via ignite
        trainer = create_supervised_trainer(base_classifier, optimizer, nn.NLLLoss(), device)
        evaluator = create_supervised_evaluator(base_classifier, {
            'accuracy': Accuracy(),
            'nll': Loss(loss)
        }, device)

        pbar = ProgressBar()
        pbar.attach(trainer)

        @trainer.on(Events.ITERATION_COMPLETED)
        def log_training_loss(engine):
            iter_in_epoch = (engine.state.iteration - 1) % len(train_loader) + 1
            if engine.state.iteration % args.print_freq == 0:
                pbar.log_message("Epoch[{}] Iteration[{}/{}] Loss: {:.2f}"
                      "".format(engine.state.epoch, iter_in_epoch, len(train_loader), engine.state.output))

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_epoch(engine):
            scheduler.step()
            evaluator.run(test_loader)

            metrics = evaluator.state.metrics
            avg_accuracy = metrics['accuracy']
            avg_nll = metrics['nll']
            
            pbar.log_message("Validation Results - Epoch: {}  Avg accuracy: {:.5f} Avg loss: {:.3f}"
                  .format(engine.state.epoch, avg_accuracy, avg_nll))
            
            log(logfilename, "Validation  - Epoch: {}  Avg accuracy: {:.5f} Avg loss: {:.3f}"
                  .format(engine.state.epoch, avg_accuracy, avg_nll))
            
            test_acc.append(avg_accuracy)
            
            if avg_accuracy >= max(test_acc):
                print("Saving the model at Epoch {:}".format(engine.state.epoch))
                torch.save({
                        'arch': args.arch,
                        'state_dict': base_classifier.state_dict(),
                        'optimizer': optimizer.state_dict(),
                }, os.path.join(args.outdir, 'checkpoint.pth.tar'))
            
            if engine.state.epoch == args.epochs:
                test_acc_list.append(max(test_acc))
                log(logfilename, "Finetuned Test Accuracy: {:.5f}".format(max(test_acc)))
                print("Finetuned Test Accuracy: ", max(test_acc))
                

        trainer.run(train_loader, args.epochs)


    log(logfilename, "This is the test accuracy list for args.round.")
    log(logfilename, str(test_acc_list))

if __name__ == "__main__":
    main()