import argparse
import sys
import os
import torch
import random
import shutil
import time
import warnings
import numpy as np
from PIL import Image
import pdb
from contextlib import redirect_stdout
import distutils
import distutils.util
import json
import importlib

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
from torch.autograd import Variable

from convert import convert, register_forward_hook


parser = argparse.ArgumentParser(description='Effect of stride testing on Imagenet')
parser.add_argument('--task', default='cifar10', choices=['imagenet', 'cifar10', 'mnist', 'imdb'], # todo: make this generic
                    help='dataset to train/evaluate on and to determine the architecture variant')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    # todo: check if model belongs to task, OR is on torchhub/huggingface/timm, OR is a path
                    help='model architecture (default: resnet18)')

parser.add_argument('--apot', default=None, type=json.loads, help='convert conv2d to APoT quantized convolution, pass argument as dict of arguments')
parser.add_argument('--deepshift', default=None, type=json.loads, help='convert conv2d to DeepShift-PS quantized convolution, pass argument as dict of arguments')
parser.add_argument('--haq', default=None, type=json.loads, help='convert conv2d and linear to HAQ quantized convolution, pass argument as dict of arguments')

parser.add_argument('--svd-decompose', default=None, type=json.loads, help='apply SVD decomposition on linear layers')
parser.add_argument('--channel-decompose', default=None, type=json.loads, help='apply channel decomposition on convolutions')
parser.add_argument('--spatial-decompose', default=None, type=json.loads, help='apply spatial decomposition on convolutions')
parser.add_argument('--depthwise-decompose', default=None, type=json.loads, help='apply depthwise decomposition on convolutions')
parser.add_argument('--tucker-decompose', default=None, type=json.loads, help='apply Tucker decomposition on convolutions')
parser.add_argument('--cp-decompose', default=None, type=json.loads, help='apply CP decomposition on convolutions')

parser.add_argument('--convup', default=None, type=json.loads, help='convert conv2d to convup, pass argument as dict of arguments')
parser.add_argument('--strideout', default=None, type=json.loads, help='add strideout to convolution, pass argument as dict of arguments')

parser.add_argument('--scale-input', default=None, type=json.loads, help='scale the input images, pass argument as dict of arguments')

parser.add_argument('--layer-start', default=0, type=int, help='index of layer to start the conversion')
parser.add_argument('--layer-end', default=-1, type=int, help='index of layer to stop the conversion')

parser.add_argument('--conversion-epoch-start', default=0, type=int, help='first epoch to apply conversion to')
parser.add_argument('--conversion-epoch-end', default=-1, type=int, help='last epoch to apply conversion to')
parser.add_argument('--conversion-epoch-step', default=1, type=int, help='epochs to skip when applying conversion')
# TODO: make --conversion-epochs be mutually exclusive the --conversion-epoch-start/end/step
parser.add_argument('--conversion-epochs', default=None, type=int, nargs='+', help='custom list of epochs to apply conversion to')

# TODO: make --image and --data mutually exclusive
parser.add_argument('-i', '--image', help='path to image')
parser.add_argument('--data-dir', default='~/datasets', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=None, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=None, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
# TODO: make lr-milestones and lr-step mutually exclusive
parser.add_argument('--lr-step-size', dest='lr_step_size', default=None, type=int,
                    help='learning rate step for StepLR schedule')
parser.add_argument('--lr-milestones', dest='lr_milestones', default=None, type=int, nargs='+', 
                    help='learning rate milestones expressed as a list of epochs for MultiStepLR schedule')
parser.add_argument('--lr-gamma', dest='lr_gamma', default=0.1, type=float,
                    help='learning rate decay')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# TODO: make --image, --evaluate, --train mutually exclusive
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', default=False, type=lambda x:bool(distutils.util.strtobool(x)), 
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
# TODO: make --cpu and --gpu mutually exclusive
parser.add_argument('--cpu', action='store_true', help='use CPU instead of GPU')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--dump-mean', dest='dump_mean', action='store_true',
                    help='log mean of each layer')

def image_loader(image_name, preprocess, device="cpu"):
    """load image, returns cuda tensor"""
    image = Image.open(image_name)
    image = preprocess(image).float().unsqueeze(0) #unsqueeze to add dimension for batch size 1
    return image.to(device)

def print_mean(m, i, o):
    print(m.__class__.__name__, ' ----> Mean: ', torch.mean(o), ' ---> std: ', torch.std(o))

best_acc1 = 0

def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)

def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    # only import the task required
    task = importlib.import_module(f"tasks.{args.task}")

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        # todo: pretrained could be name of weights, e.g., bert-case, bert-uncase, etc.
        model = task.models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = task.models.__dict__[args.arch]()

    # create epoch range
    if args.conversion_epochs is None:
        args.conversion_epochs = range(args.conversion_epoch_start, args.conversion_epoch_end+1, args.conversion_epoch_step)

    # apply conversions
    if args.svd_decompose:
        importlib.import_module(f"conversions.tensor_decomposition")
        model, _ = convert(model, torch.nn.Linear, conversions.tensor_decomposition.svd_decompose_linear, index_start=args.layer_start, index_end=args.layer_end, **args.svd_decompose)
    if args.channel_decompose:
        importlib.import_module(f"conversions.tensor_decomposition")
        model, _ = convert(model, torch.nn.Conv2d, conversions.tensor_decomposition.channel_decompose_conv, index_start=args.layer_start, index_end=args.layer_end, **args.channel_decompose)
    if args.spatial_decompose:
        importlib.import_module(f"conversions.tensor_decomposition")
        model, _ = convert(model, torch.nn.Conv2d, conversions.tensor_decomposition.spatial_decompose_conv, index_start=args.layer_start, index_end=args.layer_end, **args.spatial_decompose)
    if args.depthwise_decompose:
        importlib.import_module(f"conversions.tensor_decomposition")
        model, _ = convert(model, torch.nn.Conv2d, conversions.tensor_decomposition.depthwise_decompose_conv, index_start=args.layer_start, index_end=args.layer_end, **args.depthwise_decompose)
    if args.tucker_decompose:
        importlib.import_module(f"conversions.tensor_decomposition")
        model, _ = convert(model, torch.nn.Conv2d, conversions.tensor_decomposition.tucker_decompose_conv, index_start=args.layer_start, index_end=args.layer_end, **args.tucker_decompose)
    if args.cp_decompose:
        importlib.import_module(f"conversions.tensor_decomposition")
        model, _ = convert(model, torch.nn.Conv2d, conversions.tensor_decomposition.cp_decompose_conv_other, index_start=args.layer_start, index_end=args.layer_end, **args.cp_decompose)

    if args.apot:
        importlib.import_module(f"conversions.apot")
        model, _ = convert(model, torch.nn.Conv2d, conversions.apot.convert, index_start=args.layer_start, index_end=args.layer_end, **args.apot)
    if args.haq:
        importlib.import_module(f"conversions.haq")
        model, _ = convert(model, (torch.nn.Conv2d, torch.nn.Linear), conversions.haq.convert, index_start=args.layer_start, index_end=args.layer_end, **args.haq)
    if args.deepshift:
        importlib.import_module(f"conversions.deepshift")
        model, _ = convert(model, (torch.nn.Conv2d, torch.nn.Linear), conversions.deepshift.convert_to_shift.convert, index_start=args.layer_start, index_end=args.layer_end, **args.deepshift)

    if args.convup:
        importlib.import_module(f"conversions.convup")
        model, _ = convert(model, torch.nn.Conv2d, conversions.convup.ConvUp, index_start=args.layer_start, index_end=args.layer_end, **args.convup)
    if args.strideout:
        importlib.import_module(f"conversions.strideout")
        model, _ = convert(model, torch.nn.Conv2d, conversions.strideout.StrideOut, index_start=args.layer_start, index_end=args.layer_end, **args.strideout)

    if args.dump_mean:
        register_forward_hook(model, print_mean)

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.cpu:
        model.cpu()
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    print(model)
    print("\n press any key to continue")
    input()

    # todo: generalize device if gpu id(s) is passed
    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"

    if args.image:
        image = image_loader(args.image, task.preprocess, device) #Image filename

        model.eval()
        probabilities = model(image)
        classid = probabilities.max(1)[1].item()
        label = task.idx2label[classid]
        print("Prediction is %s with logit %.3f" %(label, probabilities[0][classid]))
        return

    # Data loading code
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        task.train_dataset(args.data_dir), batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        task.validation_dataset(args.data_dir),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # training hyperparams
    if args.epochs is not None:
        epochs = args.epochs
    else:
        epochs = task.default_epochs()

    if args.lr is not None:
        initial_lr = args.lr
    else:
        initial_lr = task.default_initial_lr()

    # define loss function (criterion) and optimizer
    criterion = task.default_criterion()
    if not args.cpu:
        criterion = criterion.cuda(args.gpu) 

    optimizer = task.default_optimizer(model, initial_lr, args.momentum, args.weight_decay)

    # define learning rate schedule
    if args.lr_milestones is not None:
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                            milestones=args.lr_milestones, 
                                                            last_epoch=args.start_epoch - 1,
                                                            gamma=args.gamma)
    elif args.lr_step_size is not None:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.gamma)
    else:
        lr_scheduler = task.default_lr_scheduler(optimizer, epochs, len(train_loader), args.start_epoch)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    if args.evaluate:
        validate(val_loader, model, criterion, args)
    else:
        for epoch in range(args.start_epoch, epochs):

            if args.distributed:
                train_sampler.set_epoch(epoch)

            # train for one epoch
            train(train_loader, task, model, criterion, optimizer, epoch, device, args)

            lr_scheduler.step()

            # evaluate on validation set
            acc1 = validate(val_loader, task, model, criterion, args)

            # remember best acc@1 and save checkpoint
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)

            if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                    and args.rank % ngpus_per_node == 0):
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'best_acc1': best_acc1,
                    'optimizer' : optimizer.state_dict(),
                }, is_best)

def train(train_loader, task, model, criterion, optimizer, epoch, device, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))
    # switch to train mode
    model.train()
    end = time.time()
    for i, batch in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        batch = task.to_device(batch, device, args.gpu)
        # optional: scale input
        # todo: clean this up and make it more generic
        # todo: perhaps add args.process_input or add this as a transformation
        if args.scale_input:
            if epoch in args.conversion_epochs:
                (images, _) = batch
                images = torch.nn.functional.interpolate(images, **args.scale_input)

        # compute output
        input, kwargs = task.get_input(batch)
        output = model(input, **kwargs)
        loss = task.get_loss(output, batch, criterion)

        # measure accuracy and record loss
        target = task.get_target(batch)
        metrics = task.get_metrics(output, target, topk=(1, 5))
        acc1, acc5 = metrics["acc1"], metrics["acc5"]
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def validate(val_loader, task, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, batch in enumerate(val_loader):
            #todo: make this generic for different tasks
            (images, target) = batch
            if args.gpu is not None and not args.cpu:
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available() and not args.cpu:
                target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            input, kwargs = task.get_input(batch)
            output = model(input, **kwargs)
            loss = task.get_loss(output, batch, criterion)

            # measure accuracy and record loss
            target = task.get_target(batch)
            metrics = task.get_metrics(output, target, topk=(1, 5))
            acc1, acc5 = metrics["acc1"], metrics["acc5"]
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

if __name__ == '__main__':
    main()
