# this file is based on code publicly available at
#   https://github.com/bearpaw/pytorch-classification
# written by Wei Yang.

import argparse
import os
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from datasets import get_dataset, DATASETS
from architectures import ARCHITECTURES, get_architecture
from torch.optim import SGD, Optimizer
from torch.optim.lr_scheduler import StepLR, MultiStepLR
import time
import datetime
from train_utils import AverageMeter, accuracy, init_logfile, log

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
parser.add_argument('--batch', default=256, type=int, metavar='N',
                    help='batchsize (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    help='initial learning rate', dest='lr')
parser.add_argument('--lr_step_size', type=int, default=30,
                    help='How often to decrease learning by gamma.')
parser.add_argument('--lr_milestones', default = [80, 120])
parser.add_argument('--gamma', type=float, default=0.1,
                    help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--noise_sd', default=0.0, type=float,
                    help="standard deviation of Gaussian noise for weight augmentation")
parser.add_argument('--gpu', default=0, type=int,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--print-freq', default=200, type=int,
                    metavar='N', help='print frequency (default: 10)')
args = parser.parse_args()


def main():
    if not os.path.exists(args.outdir):
        os.mkdir(args.outdir)

    device = torch.device("cuda")
    torch.cuda.set_device(args.gpu)
    print(args.weight_decay)
    train_dataset = get_dataset(args.dataset, 'train')
    test_dataset = get_dataset(args.dataset, 'test')
    pin_memory = (args.dataset == "imagenet")
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch,
                              num_workers=args.workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch,
                             num_workers=args.workers, pin_memory=pin_memory)

    model = get_architecture(args.arch, args.dataset, device)

    logfilename = os.path.join(args.outdir, 'train_log.txt')
    init_logfile(logfilename, "epoch\ttime\tlr\ttrain loss\ttrain acc\ttestloss\ttest acc")

    criterion = nn.NLLLoss().cuda()
    optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = MultiStepLR(optimizer, milestones=[80, 120], last_epoch=-1)


    test_acc = []
    
    # Training via ignite
    trainer = create_supervised_trainer(model, optimizer, nn.NLLLoss(), device)
    evaluator = create_supervised_evaluator(model, {
        'accuracy': Accuracy(),
        'nll': Loss(criterion)
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
        
        test_acc.append(avg_accuracy)
        
        if avg_accuracy >= max(test_acc):
            print("Saving the model at Epoch {:}".format(engine.state.epoch+args.epochs_warmup))
            torch.save({
                    'arch': args.arch,
                    'state_dict': base_classifier.state_dict(),
                    'optimizer': optimizer.state_dict(),
            }, os.path.join(args.outdir, 'checkpoint.pth.tar'))

        pbar.log_message("Validation Results - Epoch: {}  Avg accuracy: {:.4f} Avg loss: {:.2f}"
              .format(engine.state.epoch, avg_accuracy, avg_nll))

        log(logfilename, "Validation  - Epoch: {}  Avg accuracy: {:.4f} Avg loss: {:.3f}"
              .format(engine.state.epoch, avg_accuracy, avg_nll))

    trainer.run(train_loader, args.epochs)

if __name__ == "__main__":
    main()

