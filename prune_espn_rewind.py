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
from architectures import get_architecture, ARCHITECTURES
from math import ceil
from train_utils import AverageMeter, accuracy, accuracy_list, init_logfile, log
from utils import *
import utils

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
from ignite.contrib.handlers import ProgressBar

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('dataset', type=str, choices=DATASETS)
parser.add_argument('arch', type=str, choices=ARCHITECTURES)
parser.add_argument('outdir', type=str, help='folder to save model and training log)')
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
parser.add_argument('--keep_ratio', default=0.01, type=float)
parser.add_argument('--threshold', default=1e-2, type=float)
parser.add_argument('--round', default=1, type=int)
parser.add_argument('--lr_step_size', type=int, default=30,
                    help='How often to decrease learning by gamma.')
parser.add_argument('--gamma', type=float, default=0.1,
                    help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
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

    log(logfilename, "Hyperparameter List")
    log(logfilename, "Epochs: {:}".format(args.epochs))
    log(logfilename, "Learning Rate: {:}".format(args.lr))
    log(logfilename, "Alpha: {:}".format(args.alpha))
    log(logfilename, "Keep ratio: {:}".format(args.keep_ratio))
    log(logfilename, "Warmup Epochs: {:}".format(args.epochs_warmup))

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
        print("Loaded the base_classifier")
    
        criterion = nn.NLLLoss().to(device)
        optimizer = SGD(base_classifier.parameters(), lr=args.lr, momentum=args.momentum,
                        weight_decay=args.weight_decay)
    
        # Warmup training for the rewinding.
        for epoch in range(args.epochs_warmup):
            print("Warmup Training Epochs: {:}".format(epoch))
            train_loss, train_top1, train_top5 = utils.train(train_loader, 
                                                       base_classifier,
                                                       criterion,
                                                       optimizer,
                                                       epoch,
                                                       device,
                                                       print_freq=100,
                                                       display=False)

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


        criterion = nn.NLLLoss().to(device)
        optimizer = SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
                        weight_decay=0) 
          # weight_decay = 0 for training the mask.


        sparsity, total = 0, 0
        for layer in net.modules():
            if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
                boolean_list = layer.weight_mask.data > args.threshold
                sparsity += (boolean_list == 1).sum()
                total += layer.weight.numel()
                
        # Training the mask with the training set.
        for epoch in range(300):
            if epoch % 5 == 0:
                print("Current epochs: ", epoch)
                print("Sparsity: {:}".format(sparsity))
            before = time.time()
            train_loss = mask_train(train_loader, net, criterion, optimizer,
                                    epoch, device, alpha=args.alpha, display=False)
            acc = model_inference(net, test_loader, device, display=False)
            log(logfilename, "Epoch {:}, Mask Update Test Acc: {:.5}".format(epoch, acc))
            
            sparsity = 0
            total = 0
            for layer in net.modules():
                if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
                    boolean_list = layer.weight_mask.data > 1e-2
                    sparsity += (boolean_list == 1).sum()
                    total += layer.weight.numel()
            
            if sparsity <= total*args.keep_ratio:
                print("Current epochs breaking loop at {:}".format(epoch))
                break
                
                
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
        
        # Applying the mask to the original network.
        apply_prune_mask(base_classifier, keep_masks)
                
        mask_update_acc = model_inference(base_classifier, test_loader, device, display=True)
        log(logfilename, "Untrained Network Test Accuracy: {:.5}".format(mask_update_acc))
        print("Untrained Network Test Accuracy: {:.5}".format(mask_update_acc))
        

        optimizer = SGD(base_classifier.parameters(), lr=args.lr,
                        momentum=args.momentum, weight_decay=args.weight_decay)
        loss = nn.NLLLoss()
        scheduler = MultiStepLR(optimizer, milestones=[int(args.epochs*0.5)-args.epochs_warmup,
                                                       int(args.epochs*0.75)-args.epochs_warmup], 
                                last_epoch=-1)
        
        test_acc = []     # Collecting the test accuracy
        
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
            

            pbar.log_message("Validation Results - Epoch: {}  Avg accuracy: {:.3f} Avg loss: {:.3f}"
                  .format(engine.state.epoch+args.epochs_warmup, avg_accuracy, avg_nll))
            
            log(logfilename, "Validation  - Epoch: {}  Avg accuracy: {:.3f} Avg loss: {:.3f}"
                  .format(engine.state.epoch+args.epochs_warmup, avg_accuracy, avg_nll))
            
            test_acc.append(avg_accuracy)
            
            
            if avg_accuracy >= max(test_acc):
                print("Saving the model at Epoch {:}".format(engine.state.epoch+args.epochs_warmup))
                torch.save({
                        'arch': args.arch,
                        'state_dict': base_classifier.state_dict(),
                        'optimizer': optimizer.state_dict(),
                }, os.path.join(args.outdir, 'checkpoint.pth.tar'))
            
            if engine.state.epoch+args.epochs_warmup == args.epochs:
                test_acc_list.append(max(test_acc))
                log(logfilename, "Finetuned Test Accuracy: {:.5f}".format(max(test_acc)))
                print("Finetuned Test Accuracy: ", max(test_acc))
                

        trainer.run(train_loader, args.epochs-args.epochs_warmup)


    log(logfilename, "This is the test accuracy list for args.round.")
    log(logfilename, str(test_acc_list))

    
if __name__ == "__main__":
    main()