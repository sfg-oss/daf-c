#-*-coding:utf-8-*-
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
import argparse
import time
import re
import os
import daf
from datasets.dataset import Data
import cfg
import log
import cv2



def MAE_multilayer(inputs, targets, cuda=False, balance=1.1):
    diff = inputs - targets
    squared_diff = diff ** 2
    loss = squared_diff.sum()
    return loss/12


def adjust_learning_rate(optimizer, steps, step_size, gamma=0.1, logger=None):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""

    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * gamma
        if logger:
            logger.info('%s: %s' % (param_group['name'], param_group['lr']))

def cross_entropy_loss2d(inputs, targets, cuda=False, balance=1.1):

    n, c, h, w = inputs.size()

    weights = np.zeros((n, c, h, w))
    for i in range(n):
        t = targets[i, :, :, :].cpu().data.numpy()
        pos = (t == 1).sum()
        neg = (t == 0).sum()
        valid = neg + pos+1
        weights[i, t == 1] = neg * 1. / valid
        weights[i, t == 0] = pos * balance / valid
    weights = torch.Tensor(weights)


    if cuda:
        weights = weights.cuda()
    inputs = torch.sigmoid(inputs)

    loss=nn.BCELoss(weights,reduction='sum')(inputs,targets)


    return loss

def re_Dice_Loss(inputs, targets, cuda=False, balance=1.1):
    n, c, h, w = inputs.size()
    smooth=1
    inputs = torch.sigmoid(inputs)  # F.sigmoid(inputs)
    input_flat=inputs.view(-1)
    target_flat=targets.view(-1)
    intersecion=input_flat*target_flat
    unionsection=input_flat.pow(2).sum()+target_flat.pow(2).sum()+smooth
    loss=unionsection/(2*intersecion.sum()+smooth)
    loss=loss.sum()
    return loss

def train(model, args):
    data_root = cfg.config[args.dataset]['data_root']
    data_lst = cfg.config[args.dataset]['data_lst']

    mean_bgr = np.array(cfg.config[args.dataset]['mean_bgr'])
    yita = args.yita if args.yita else cfg.config[args.dataset]['yita']
    crop_size = args.crop_size
    print(f"data_root{data_root}")
    train_img = Data(data_root, data_lst, yita, mean_bgr=mean_bgr, crop_size=crop_size)
    print(train_img.scale)
    trainloader = torch.utils.data.DataLoader(train_img,
        batch_size=args.batch_size, shuffle=True, num_workers=4)  #num_workers=5
    print("加载数据")

    logger = args.logger
    params = []

    optimizer = torch.optim.SGD(params, momentum=args.momentum,
        lr=args.base_lr, weight_decay=args.weight_decay)

    start_step = 1
    mean_loss = []
    cur = 0
    pos = 0
    data_iter = iter(trainloader)
    iter_per_epoch = len(trainloader)
    logger.info('*'*40)
    logger.info('train images in all are %d ' % (iter_per_epoch/args.iter_size))
    logger.info('*'*40)
    print(f"iter_per_epoch{iter_per_epoch/args.iter_size}")
    for param_group in optimizer.param_groups:

        if logger:
            logger.info('%s: %s' % (param_group['name'], param_group['lr']))
    start_time = time.time()
    if args.cuda:
        model.cuda()
    if args.resume:
        logger.info('resume from %s' % args.resume)

        state = torch.load(args.resume)

        start_step = state['step']
        optimizer.load_state_dict(state['solver'])

        model.load_state_dict(state['param'])
    model.train()

    batch_size = args.iter_size * args.batch_size
    for step in range(start_step, args.max_iter + 1):
        print(f"第{step}/{args.max_iter}轮 ")
        optimizer.zero_grad()
        batch_loss = 0
        for i in range(args.iter_size):
            if cur == iter_per_epoch:
                cur = 0
                data_iter = iter(trainloader)
            images, labels = next(data_iter)
            if args.cuda:
                images, labels = images.cuda(), labels.cuda()
            images, labels = Variable(images), Variable(labels)
            out = model(images)

            la=images[0]
            la = la.permute((1, 2, 0))

            la = la + torch.tensor([104.00699, 116.66877, 122.67892]).cuda()
            la = la[:, :, [2, 1, 0]]

            la = la.cpu()
            la = la.numpy()
            la = cv2.blur(la, (2, 2)).astype(np.uint8)

            gray_image = cv2.cvtColor(la, cv2.COLOR_BGR2GRAY)
            la=cv2.Canny(gray_image,20,80)

            la = torch.from_numpy(la).cuda()

            la = la.unsqueeze(0)
            canny1= la.unsqueeze(0)/255
            canny=(out[-1]*canny1).detach()
            canny=torch.sigmoid(canny)

            canny = torch.where(canny > 0.5, torch.tensor(1, dtype=canny.dtype, device=canny.device), canny)

            cannyl=[]
            for i in range(0,len(out)-1):
                canny_tem=(out[i]*canny1).detach()
                canny_tem = torch.sigmoid(canny_tem)
                canny_tem = torch.where(canny_tem >0.5, torch.tensor(1), canny_tem)

                cannyl.append(canny_tem)
            loss = 0
            loss +=(args.fuse_weight*cross_entropy_loss2d(out[-1], labels, args.cuda, args.balance)/batch_size
             +((args.reDice_weight*re_Dice_Loss(out[-1], labels, args.cuda, args.balance)/batch_size))
                    )
            # #
            # print()

            loss +=(args.fuse_weight * cross_entropy_loss2d(out[-1], canny, args.cuda, args.balance) / batch_size)
            loss +=((args.reDice_weight * re_Dice_Loss(out[-1], canny, args.cuda, args.balance) / batch_size))



            loss.backward()
            batch_loss +=loss.item()
            cur += 1


        optimizer.step()
        if len(mean_loss) < args.average_loss:
            mean_loss.append(batch_loss)
        else:
            mean_loss[pos] = batch_loss
            pos = (pos + 1) % args.average_loss
        if step % args.step_size == 0:
            adjust_learning_rate(optimizer, step, args.step_size, args.gamma)
        if step % args.snapshots == 0:
            torch.save(model.state_dict(), '%s/daf_%d.pth' % (args.param_dir, step))
            state = {'step': step+1,'param':model.state_dict(),'solver':optimizer.state_dict()}
            # torch.save(state, '%s/daf_%d.pth.tar' % (args.param_dir, step))
        if step % args.display == 0:
            tm = time.time() - start_time
            logger.info('iter: %d, lr: %e, loss: %f, time using: %f(%fs/iter)' % (step,
                optimizer.param_groups[0]['lr'], np.mean(mean_loss), tm, tm/args.display))
            start_time = time.time()


def main():
    # print("ifgeubggfbvreuvbgu")
    args = parse_args()
    logger = log.get_logger(args.log)
    args.logger = logger
    logger.info('*'*80)
    logger.info('the args are the below')
    logger.info('*'*80)
    for x in args.__dict__:
        logger.info(x+','+str(args.__dict__[x]))
    logger.info(cfg.config[args.dataset])
    logger.info('*'*80)
    # print("iwsugfiwgefvuy")
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    if not os.path.exists(args.param_dir):
        os.mkdir(args.param_dir)
    torch.manual_seed(int(time.time()))
    model = daf.daf(pretrain=args.pretrain, logger=logger)
    # model=resnet_my.resnet_daf(pretrain=args.pretrain, logger=logger)
    if args.complete_pretrain:
        model.load_state_dict(torch.load(args.complete_pretrain))
    # logger.info(model)
    train(model, args)

resdir='param_all/'

snapshots=15000
itersize=10
cuda="1"
complete_model=None
dataset="bsds500"
base_lr=1e-6

print(f'开始训练{dataset}')



def parse_args():
    parser = argparse.ArgumentParser(description='Train daf for different args')
    parser.add_argument('-d', '--dataset', type=str, choices=cfg.config.keys(),
        default=dataset, help='The dataset to train')
    parser.add_argument('--param-dir', type=str, default=resdir,
        help='the directory to store the params')
    parser.add_argument('--lr', dest='base_lr', type=float, default=base_lr,
        help='the base learning rate of model')
    parser.add_argument('-m', '--momentum', type=float, default=0.9,
        help='the momentum')
    parser.add_argument('-c', '--cuda', action='store_true',default=True,   #usage: -c,not use in default
        help='whether use gpu to train network')
    parser.add_argument('-g', '--gpu', type=str, default=cuda,
        help='the gpu id to train net')
    parser.add_argument('--weight-decay', type=float, default=0.0002,
        help='the weight_decay of net')
    parser.add_argument('-r', '--resume', type=str, default=None,
        help='whether resume from some, default is None')
    parser.add_argument('-p', '--pretrain', type=str, default="vgg16.pth",
        help='init net from pretrained model default is None')
    parser.add_argument('--max-iter', type=int, default=snapshots,
        help='max iters to train network, default is 40000')
    parser.add_argument('--iter-size', type=int, default=itersize,
        help='iter size equal to the batch size, default 10')
    parser.add_argument('--average-loss', type=int, default=50,
        help='smoothed loss, default is 50')
    parser.add_argument('-s', '--snapshots', type=int, default=snapshots,  #1000
        help='how many iters to store the params, default is 1000')
    parser.add_argument('--step-size', type=int, default=5000,
        help='the number of iters to decrease the learning rate, default is 10000')
    parser.add_argument('--display', type=int, default=20,
        help='how many iters display one time, default is 20')
    parser.add_argument('-b', '--balance', type=float, default=1.1,
        help='the parameter to balance the neg and pos, default is 1.1')
    parser.add_argument('-l', '--log', type=str, default='log2.txt',
        help='the file to store log, default is log.txt')
    parser.add_argument('-k', type=int, default=1,
        help='the k-th split set of multicue')
    parser.add_argument('--batch-size', type=int, default=1,
        help='batch size of one iteration, default 1')
    parser.add_argument('--crop-size', type=int, default=None,
        help='the size of image to crop, default not crop')
    parser.add_argument('--yita', type=float, default=0.5,
        help='the param to operate gt, default is data in the config file')

    parser.add_argument('--complete-pretrain', type=str, default=complete_model,
        help='finetune on the complete_pretrain, default None')
    parser.add_argument('--side-weight', type=float, default=0.2,
        help='the loss weight of sideout, default 0.5')
    parser.add_argument('--fuse-weight', type=float, default=1.1,
        help='the loss weight of fuse, default 1.1')
    parser.add_argument('--reDice-weight', type=float, default=100,
                        help='the loss weight of fuse, default 10')
    parser.add_argument('--gamma', type=float, default=0.1,
        help='the decay of learning rate, default 0.1')
    return parser.parse_args()

if __name__ == '__main__':
    print("开始")
    main()



