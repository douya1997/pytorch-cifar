'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

import os
import argparse
import logging
import time
import shutil

from models import *
from utils import create_logger, adjust_learning_rate, AverageMeter, accuracy, to_python_float


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--ckpt', default='experiment/resnet18', type=str)
parser.add_argument('--max_epoch', default=350, type=int)
parser.add_argument('--lr_steps', nargs='+', type=int)
parser.add_argument('--print_freq', default=20, type=int)

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

print(args)
# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# mkdir
if not os.path.exists(os.path.join(args.ckpt, 'checkpoints')):
    os.makedirs(os.path.join(args.ckpt, 'checkpoints'))
if not os.path.exists(os.path.join(args.ckpt, 'logs')):
    os.makedirs(os.path.join(args.ckpt, 'logs'))

# logger
logger = create_logger('global_logger', '{}/logs/{}.txt'.format(args.ckpt, time.time()))


# Model
print('==> Building model..')
# net = VGG('VGG19')
net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.t7')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)


# Training
def train(epoch):
    freq = args.print_freq
    losses = AverageMeter(freq)
    top1 = AverageMeter(freq)
    top5 = AverageMeter(freq)

    print('\nEpoch: %d' % epoch)
    net.train()
    lr = adjust_learning_rate(optimizer, epoch, args)

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
         
        prec1, prec5 = accuracy(outputs.data, targets, topk=(1,5))

        losses.update(to_python_float(loss))
        top1.update(to_python_float(prec1))
        top5.update(to_python_float(prec5))   

        if batch_idx % args.print_freq == 0 and batch_idx > 1:
             logger = logging.getLogger('global_logger')

             logger.info('Epoch: [{0}][{1}/{2}]\t'
                       'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                       'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                       'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'
                       'LR {lr:.3f}'.format(
                        epoch, batch_idx, len(trainloader),
                        loss=losses, top1=top1, top5=top5, lr=lr))
             niter = epoch*len(trainloader)+batch_idx


def test(epoch):
    global best_acc
    net.eval()

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            prec1, prec5 = accuracy(outputs.data, targets, topk=(1,5))
            losses.update(to_python_float(loss), inputs.size(0))
            top1.update(to_python_float(prec1), inputs.size(0))
            top5.update(to_python_float(prec5), inputs.size(0))   

            
            if batch_idx % args.print_freq == 0 and batch_idx > 1:
                logger = logging.getLogger('global_logger')

                logger.info('Epoch: [{0}][{1}/{2}]\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'
                       .format(epoch, batch_idx, len(trainloader),
                       loss=losses, top1=top1, top5=top5))

                niter = epoch*len(trainloader)+batch_idx

        logger.info(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
        .format(top1=top1, top5=top5))

    # Save checkpoint.
    acc = top1.avg
    print('Saving..')
    state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
    }
    filename =  os.path.join(args.ckpt, 'checkpoints', 'checkpoint.pth.tar')
    bestfile =  os.path.join(args.ckpt, 'checkpoints', 'model_best.pth.tar')
    torch.save(state,filename)
    if acc > best_acc:
        best_acc = acc
        shutil.copyfile(filename,bestfile)


for epoch in range(start_epoch, args.max_epoch):
    train(epoch)
    test(epoch)
