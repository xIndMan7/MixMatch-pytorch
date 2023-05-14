from __future__ import print_function

import argparse
import os
import shutil
import time
import random

import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.nn.functional as F
from torchvision.models import vgg19, resnet50 as res50_pt, resnet18 as res18_pt, densenet121, convnext_tiny, wide_resnet50_2
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152
from ray_models import ResNet18, ResNet34, ResNet50, ResNet101

import sys
if '/serenity/data/ppml/' not in sys.path:
    sys.path.append('/serenity/data/ppml/')

# if '/nethome/dsanyal7/ppml/ppml_model_serving/src/clockwork/' not in sys.path:
#     sys.path.append('/nethome/dsanyal7/ppml/')

import models.wideresnet as models
import dataset.cifar10 as dataset
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
from tensorboardX import SummaryWriter
from cifar10_models.resnet import resnet18 as resnet18_m, resnet34 as resnet34_m, resnet50 as resnet50_m, resnet152 as resnet152_m
# from cifar10_models.resnet import resnet18, resnet34, resnet50
from cifar10_models.vgg import vgg11_bn, vgg19_bn
from ppml_model_serving.src.model_serving.model_server import *
from clockwork.clockwork import *

import sys
import wandb

import importlib  
foobar = importlib.import_module("pytorch-cifar.models")


dispatcher = {18: resnet18, 34: resnet34, 50: resnet50, 101: resnet101, 152: resnet152}
mixmatch_dispatcher = {18: resnet18_m, 34: resnet34_m, 50: resnet50_m, 152: resnet152_m}

parser = argparse.ArgumentParser(description='PyTorch MixMatch Training')
# Optimization options
parser.add_argument('--direct', default=False, type=bool, metavar='If Direct Model Zoo',
                    help='run model zoo', action=argparse.BooleanOptionalAction)
parser.add_argument('--zoo', default=False, type=bool, metavar='If zoo',
                    help='run model zoo', action=argparse.BooleanOptionalAction)
parser.add_argument('--clockwork', default=False, type=bool, metavar='If clockwork',
                    help='run clockwork model zoo', action=argparse.BooleanOptionalAction)
parser.add_argument('--server', default=False, type=bool, metavar='If zoo',
                    help='use inference', action=argparse.BooleanOptionalAction)
parser.add_argument('--zoo_victim', default=0, type=int, metavar='which model in zoo',
                    help='model zoo')
# parser.add_argument('--oracle-server', default="", type=str, metavar='',
#                     help='')
parser.add_argument('--fingerprinting', default=False, type=bool, metavar='',
                    help='', action=argparse.BooleanOptionalAction)
parser.add_argument('--epochs', default=1024, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--batch-size', default=64, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.002, type=float,
                    metavar='LR', help='initial learning rate')

parser.add_argument('--om', '--oracle_model', default="", type=str,
                    metavar='oracle_model', help='oracle_model')

parser.add_argument('--om_r', '--oracle_model_resnet', default=18, type=int,
                    metavar='resnet type', help='which oracle resnet type')

parser.add_argument('--am_r', '--attack_model_resnet', default=18, type=int,
                    metavar='resnet type', help='which attack resnet type')
# Checkpoints
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# Miscs
parser.add_argument('--manualSeed', type=int, default=0, help='manual seed')
#Device options
parser.add_argument('--gpu', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
#Method options
parser.add_argument('--n-labeled', type=int, default=250,
                        help='Number of labeled data')
parser.add_argument('--train-iteration', type=int, default=1024,
                        help='Number of iteration per epoch')
parser.add_argument('--out', default='result',
                        help='Directory to output the result')
parser.add_argument('--alpha', default=0.75, type=float)
parser.add_argument('--lambda-u', default=75, type=float)
parser.add_argument('--T', default=0.5, type=float)
parser.add_argument('--ema-decay', default=0.999, type=float)


args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}
print(args)

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
np.random.seed(args.manualSeed)

best_acc = 0  # best test accuracy

def main():
    global best_acc

    if not os.path.isdir(args.out):
        mkdir_p(args.out)

    # Data
    print(f'==> Preparing cifar10')
    transform_train = transforms.Compose([
        dataset.RandomPadandCrop(32),
        dataset.RandomFlip(),
        dataset.ToTensor(),
    ])

    transform_val = transforms.Compose([
        dataset.ToTensor(),
    ])

    server = ModelServer()
    model_zoo = {}

    acc_lat_dict = {}

    if args.zoo is False:
        OracleModel = dispatcher[args.om_r]()
        if args.om:
            print("Oracle File Path: ", args.om)
            OracleModel.load_state_dict(torch.load(args.om))
        model_zoo = {f"resnet{args.om_r}": OracleModel}
    else:
        # OracleModel = []
        # vgg = vgg11_bn()
        # vgg19 = vgg19_bn()
        # res18 = resnet18()
        # res34 = resnet34()
        # res50 = resnet50()
        # OracleModel.append(vgg)
        # OracleModel.append(res18)
        # OracleModel.append(res34)
        # OracleModel.append(res50)
        # OracleModel[0].load_state_dict(torch.load("models/pretrained_vgg/vgg11_bn.pt"))
        # OracleModel[1].load_state_dict(torch.load("models/pretrained_resnet/model_9.pth"))
        # OracleModel[2].load_state_dict(torch.load("models/pretrained_resnet/resnet34_less_acc.pth"))
        # OracleModel[3].load_state_dict(torch.load("models/pretrained_resnet/resnet50.pt"))

        # r50_m = res50_pt().float()
        # # r50_m.cuda()
        # r50_m.load_state_dict(torch.load("../resnet50/model_6.pth"))
        # # r50_m.eval()

        # # vgg19 = vgg11_bn().to(device)
        # # vgg19.cuda()
        # # vgg19.load_state_dict(torch.load("../models/pretrained_vgg/vgg11_bn.pt"))
        # # vgg19.eval()

        # d121 = densenet121().float()
        # # d121.cuda()
        # d121.load_state_dict(torch.load("../densenet121/model_8.pth"))
        # # d121.eval()

        # convn = convnext_tiny().float()
        # # convn.cuda()
        # convn.load_state_dict(torch.load("../convnext/model_9.pth"))
        # # convn.eval()

        # w50_2 = wide_resnet50_2().float()
        # # w50_2.cuda()
        # w50_2.load_state_dict(torch.load("../widerresnet50/model_9.pth"))
        # # w50_2.eval()

        if args.clockwork:
            print("clockwork enabled")
            
            models_from_file = []
            # clockwork_accuracy=[0.6631, 0.6991, 0.7068, 0.0005]
            # clockwork_latency=[10, 12, 18, 13]
            clockwork_accuracy = []
            clockwork_latency = []
            my_file = open("models.txt", "r")
            data = my_file.read()
            models_info_from_file = data.split("\n")
            for m in models_info_from_file:
                info = m.split(" ")
                model_name, acc, lat = info[0], info[1], info[2]
                clockwork_accuracy.append(float(acc))
                clockwork_latency.append(int(lat))
                models_from_file.append(model_name)
            print("Models being loaded to Clockwork", models_from_file)

            init_request = InitializeRequest(
                #### ALERT: RESNET18 not loaded
                models_to_load=models_from_file,
                models_accuracy=clockwork_accuracy,
                models_latency_us=clockwork_latency
            )
            print("Load Requests Sending....")
            init_request.send()
            init_response = InitializeResponse.receive()
            print("InitializeResponse success:", init_response.success)
            
            model_zoo = None

            # print("Oracle File Path: ", args.om)
            OracleModel = resnet50()
            OracleModel.load_state_dict(torch.load("../resnet50/model_8.pth"))

        
        else:
            OracleModel = []

            ##### MID-ZOO ####
            # r18_m = resnet18().float()
            # r18_m.load_state_dict(torch.load("../resnet18/model_0.pth"))
            # r18_m.eval()

            # r34_m = resnet34().float()
            # r34_m.load_state_dict(torch.load("../resnet34/model_2.pth"))
            # r34_m.eval()

            # r50_m = resnet50().float()
            # r50_m.load_state_dict(torch.load("../resnet50/model_8.pth"))
            # r50_m.eval()


            # r101_m = resnet101().float()
            # r101_m.load_state_dict(torch.load("../resnet101/model_2.pth"))
            # r101_m.eval()

            # r152_m = resnet152().float()
            # r152_m.load_state_dict(torch.load("../resnet152/model_4.pth"))
            # r152_m.eval()

            #### HIGH-ZOO ###
            # r18_m = resnet18_m().float()
            # r18_m.load_state_dict(torch.load("../high_zoo/resnet18.pt"))
            # r18_m.eval()


            # r34_m = resnet34_m().float()
            # r34_m.load_state_dict(torch.load("../high_zoo/resnet34.pt"))
            # r34_m.eval()


            # r50_m = resnet50_m().float()
            # r50_m.load_state_dict(torch.load("../high_zoo/resnet50.pt"))
            # r50_m.eval()


            # r101_m = foobar.ResNet101().float()
            # t_101_d = torch.load("../pytorch-cifar/checkpoint/ckpt.pth")["net"]
            # r101_dict_n = {}
            # for k in t_101_d.keys():
            #     if "module." in k:
            #         new_k = k.replace("module.", "")
            #         r101_dict_n[new_k] = t_101_d[k]
            # r101_m.load_state_dict(r101_dict_n)
            # r101_m.eval()


            # r152_m = foobar.ResNet152().float()
            # t_152_d = torch.load("../pytorch-cifar/checkpoint/ckpt1.pth")["net"]
            # r152_dict_n = {}
            # for k in t_152_d.keys():
            #     if "module." in k:
            #         new_k = k.replace("module.", "")
            #         r152_dict_n[new_k] = t_152_d[k]
            # r152_m.load_state_dict(r152_dict_n)
            # r152_m.eval()


            #### NEW SPREAD OUT ZOO
            r18_m = ResNet18().float()
            # r18_m.load_state_dict(torch.load("../resnet18/model_0.pth"))
            r18_m.load_state_dict(torch.load("/serenity/data/ppml/ppml_model_serving/models/resnet18_model_acc_76_7.pth", map_location='cpu'))
            r18_m.eval()

            r34_m = ResNet34().float()
            # r34_m.load_state_dict(torch.load("../resnet34/model_2.pth"))
            r34_m.load_state_dict(torch.load("/serenity/data/ppml/ppml_model_serving/models/resnet34_model_acc_80_1.pth", map_location='cpu'))
            r34_m.eval()

            r50_m = ResNet50().float()
            # r50_m.load_state_dict(torch.load("../resnet50/model_8.pth"))
            r50_m.load_state_dict(torch.load("/serenity/data/ppml/ppml_model_serving/models/resnet50_model_acc_86_7.pth", map_location='cpu'))
            r50_m.eval()

            r101_m = ResNet101().float()
            # r50_m.load_state_dict(torch.load("../resnet50/model_8.pth"))
            r101_m.load_state_dict(torch.load("/serenity/data/ppml/ppml_model_serving/models/resnet101_model_acc_89_9.pth", map_location='cpu'))
            r101_m.eval()

            r152_m = foobar.ResNet152().float()
            t_152_d = torch.load("../pytorch-cifar/checkpoint/ckpt1.pth")["net"]
            r152_dict_n = {}
            for k in t_152_d.keys():
                if "module." in k:
                    new_k = k.replace("module.", "")
                    r152_dict_n[new_k] = t_152_d[k]
            r152_m.load_state_dict(r152_dict_n)
            r152_m.eval()


            model_zoo = {
            # #     "r18": r18_m,
            #     "r50": r50_m,
            # #     "vgg19": vgg19,
            #     "densenet121": d121,
            #     "convnext": convn,
            #     "widerresnet": w50_2
                "r18": r18_m,
                "r34": r34_m,
                "r50": r50_m,
                "r101": r101_m,
                "r152": r152_m,
            }

            for k, v in model_zoo.items():
                OracleModel.append(v)
    
    print("Labelled: ", args.n_labeled)
    train_labeled_set, train_unlabeled_set, val_set, test_set, oracle_train_dataset, oracle_actual_dataset, oracle_val_dataset, oracle_test_dataset = dataset.get_cifar10('./data', args.n_labeled, 
                                                        oracle_model=OracleModel, transform_train=transform_train, transform_val=transform_val, use_cuda=use_cuda, which_model=args.zoo_victim, server=args.server,
                                                        model_zoo=model_zoo, fingerprinting=args.fingerprinting, clockwork=args.clockwork, clockwork_acc=clockwork_accuracy, clockwork_lat=clockwork_latency,
                                                        direct=args.direct)
    labeled_trainloader = data.DataLoader(train_labeled_set, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    unlabeled_trainloader = data.DataLoader(train_unlabeled_set, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    # oracle_trainloader = data.DataLoader(train_unlabeled_set, model=args.om, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_loader = data.DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    oracle_train_loader = data.DataLoader(oracle_train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    oracle_train_actual_loader = data.DataLoader(oracle_actual_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    oracle_val_loader = data.DataLoader(oracle_val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    oracle_test_loader = data.DataLoader(oracle_test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # Model

    def create_model(ema=False, args=None):
        # model = models.WideResNet(num_classes=10)
        if args:
            model = mixmatch_dispatcher[args.am_r](pretrained=False)
            # model = resnet18_m(pretrained=False)
        else:
            model = resnet18(pretrained=False)
        model = model.cuda()

        if ema:
            for param in model.parameters():
                param.detach_()

        return model

    model = create_model(args=args)
    ema_model = create_model(ema=True, args=args)

    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    train_criterion = SemiLoss()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    ema_optimizer= WeightEMA(model, ema_model, alpha=args.ema_decay)
    start_epoch = 0

    print(args)

    wandb.init(
    # set the wandb project where this run will be logged
        project="ppml_proj",
        
        # track hyperparameters and run metadata
        config={
            "learning_rate": args.lr,
            "architecture": "Resnet",
            "dataset": "CIFAR-10",
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "start_epochs": args.start_epoch,

            "iteration": args.train_iteration,
            "labelled_datapoints": args.n_labeled,

            "oracle_resnet": args.om_r,
            "attack_resnet": args.am_r,
            "alpha": args.alpha,
            "lambda_u": args.lambda_u,
            "T": args.lambda_u,
            "ema_decay": args.ema_decay,
            "model_zoo": args.zoo,
        }
    )

    # Resume
    title = 'noisy-cifar-10'
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.out = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        ema_model.load_state_dict(checkpoint['ema_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger = Logger(os.path.join(args.out, str(args.om_r)+'o_'+str(args.am_r)+'_'+str(args.n_labeled)+'_log.txt'), title=title, resume=True)
    else:
        logger = Logger(os.path.join(args.out, str(args.om_r)+'o_'+str(args.am_r)+'_'+str(args.n_labeled)+'_log.txt'), title=title)
        logger.set_names(['Train Loss', 'Train Loss X', 'Train Loss U',  'Valid Loss', 'Valid Acc.', 'Test Loss', 'Test Acc.', 'Train Fid', 'Valid Fid', 'Test Fid'])

    writer = SummaryWriter(args.out)
    step = 0
    test_accs = []
    # Train and val
    for epoch in range(start_epoch, args.epochs):

        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))

        train_loss, train_loss_x, train_loss_u = train(oracle_train_loader, unlabeled_trainloader, model, optimizer, ema_optimizer, train_criterion, epoch, use_cuda)
        _, train_acc = validate(labeled_trainloader, ema_model, criterion, epoch, use_cuda, mode='Train Stats')
        val_loss, val_acc = validate(val_loader, ema_model, criterion, epoch, use_cuda, mode='Valid Stats')
        test_loss, test_acc = validate(test_loader, ema_model, criterion, epoch, use_cuda, mode='Test Stats ')

        _, train_fidelity = validate(oracle_train_actual_loader, ema_model, criterion, epoch, use_cuda, mode='Train Fidelity Stats')
        _, val_fidelity = validate(oracle_val_loader, ema_model, criterion, epoch, use_cuda, mode='Valid Fidelity Stats')
        _, test_fidelity = validate(oracle_test_loader, ema_model, criterion, epoch, use_cuda, mode='Test Fidelity Stats ')

        step = args.train_iteration * (epoch + 1)

        writer.add_scalar('losses/train_loss', train_loss, step)
        writer.add_scalar('losses/valid_loss', val_loss, step)
        writer.add_scalar('losses/test_loss', test_loss, step)

        writer.add_scalar('accuracy/train_acc', train_acc, step)
        writer.add_scalar('accuracy/val_acc', val_acc, step)
        writer.add_scalar('accuracy/test_acc', test_acc, step)

        writer.add_scalar('fidelity/train_fid', train_fidelity, step)
        writer.add_scalar('fidelity/val_fid', val_fidelity, step)
        writer.add_scalar('fidelity/test_fid', test_fidelity, step)

        # append logger file
        logger.append([train_loss, train_loss_x, train_loss_u, val_loss, val_acc, test_loss, test_acc, train_fidelity, val_fidelity, test_fidelity])
        wandb.log({"train_loss": train_loss, "train_loss_labelled": train_loss_x, "train_loss_unlabelled": train_loss_u, "validation_loss": val_loss, "validation_accuracy": val_acc, "test_loss": test_loss, "test_acc": test_acc, "fidelity_train": train_fidelity, "fidelity_val": val_fidelity, "fidelity_test": test_fidelity, "train_acc": train_acc})

        # save model
        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)
        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'ema_state_dict': ema_model.state_dict(),
                'acc': val_acc,
                'best_acc': best_acc,
                'optimizer' : optimizer.state_dict(),
            }, is_best)
        test_accs.append(test_acc)
    logger.close()
    writer.close()

    print('Best acc:')
    print(best_acc)

    print('Mean acc:')
    print(np.mean(test_accs[-20:]))


def train(labeled_trainloader, unlabeled_trainloader, model, optimizer, ema_optimizer, criterion, epoch, use_cuda):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_x = AverageMeter()
    losses_u = AverageMeter()
    ws = AverageMeter()
    end = time.time()

    bar = Bar('Training', max=args.train_iteration)
    labeled_train_iter = iter(labeled_trainloader)
    unlabeled_train_iter = iter(unlabeled_trainloader)

    model.train()
    for batch_idx in range(args.train_iteration):
        try:
            inputs_x, targets_x = next(labeled_train_iter)
        except:
            labeled_train_iter = iter(labeled_trainloader)
            inputs_x, targets_x = next(labeled_train_iter)

        try:
            (inputs_u, inputs_u2), _ = next(unlabeled_train_iter)
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            (inputs_u, inputs_u2), _ = next(unlabeled_train_iter)

        # measure data loading time
        data_time.update(time.time() - end)

        batch_size = inputs_x.size(0)

        # Transform label to one-hot
        targets_x = torch.zeros(batch_size, 10).scatter_(1, targets_x.view(-1,1).long(), 1)

        if use_cuda:
            inputs_x, targets_x = inputs_x.cuda(), targets_x.cuda(non_blocking=True)
            inputs_u = inputs_u.cuda()
            inputs_u2 = inputs_u2.cuda()


        with torch.no_grad():
            # compute guessed labels of unlabel samples
            outputs_u = model(inputs_u)
            outputs_u2 = model(inputs_u2)
            p = (torch.softmax(outputs_u, dim=1) + torch.softmax(outputs_u2, dim=1)) / 2
            pt = p**(1/args.T)
            targets_u = pt / pt.sum(dim=1, keepdim=True)
            targets_u = targets_u.detach()

        # mixup
        all_inputs = torch.cat([inputs_x, inputs_u, inputs_u2], dim=0)
        all_targets = torch.cat([targets_x, targets_u, targets_u], dim=0)

        l = np.random.beta(args.alpha, args.alpha)

        l = max(l, 1-l)

        idx = torch.randperm(all_inputs.size(0))

        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]

        mixed_input = l * input_a + (1 - l) * input_b
        mixed_target = l * target_a + (1 - l) * target_b

        # interleave labeled and unlabed samples between batches to get correct batchnorm calculation 
        mixed_input = list(torch.split(mixed_input, batch_size))
        mixed_input = interleave(mixed_input, batch_size)

        logits = [model(mixed_input[0])]
        for input in mixed_input[1:]:
            logits.append(model(input))

        # put interleaved samples back
        logits = interleave(logits, batch_size)
        logits_x = logits[0]
        logits_u = torch.cat(logits[1:], dim=0)

        Lx, Lu, w = criterion(logits_x, mixed_target[:batch_size], logits_u, mixed_target[batch_size:], epoch+batch_idx/args.train_iteration)

        loss = Lx + w * Lu

        # record loss
        losses.update(loss.item(), inputs_x.size(0))
        losses_x.update(Lx.item(), inputs_x.size(0))
        losses_u.update(Lu.item(), inputs_x.size(0))
        ws.update(w, inputs_x.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ema_optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | Loss_x: {loss_x:.4f} | Loss_u: {loss_u:.4f} | W: {w:.4f}'.format(
                    batch=batch_idx + 1,
                    size=args.train_iteration,
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    loss_x=losses_x.avg,
                    loss_u=losses_u.avg,
                    w=ws.avg,
                    )
        bar.next()
    bar.finish()

    return (losses.avg, losses_x.avg, losses_u.avg,)

def validate(valloader, model, criterion, epoch, use_cuda, mode):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    bar = Bar(f'{mode}', max=len(valloader))
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(valloader):
            # measure data loading time
            data_time.update(time.time() - end)

            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
            # compute output
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                        batch=batch_idx + 1,
                        size=len(valloader),
                        data=data_time.avg,
                        bt=batch_time.avg,
                        total=bar.elapsed_td,
                        eta=bar.eta_td,
                        loss=losses.avg,
                        top1=top1.avg,
                        top5=top5.avg,
                        )
            bar.next()
        bar.finish()
    return (losses.avg, top1.avg)

def save_checkpoint(state, is_best, checkpoint=args.out, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))

def linear_rampup(current, rampup_length=args.epochs):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current / rampup_length, 0.0, 1.0)
        return float(current)

class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch):
        probs_u = torch.softmax(outputs_u, dim=1)

        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u)**2)

        return Lx, Lu, args.lambda_u * linear_rampup(epoch)

class WeightEMA(object):
    def __init__(self, model, ema_model, alpha=0.999):
        self.model = model
        self.ema_model = ema_model
        self.alpha = alpha
        self.params = list(model.state_dict().values())
        self.ema_params = list(ema_model.state_dict().values())
        self.wd = 0.02 * args.lr

        for param, ema_param in zip(self.params, self.ema_params):
            param.data.copy_(ema_param.data)

    def step(self):
        one_minus_alpha = 1.0 - self.alpha
        for param, ema_param in zip(self.params, self.ema_params):
            if ema_param.dtype==torch.float32:
                ema_param.mul_(self.alpha)
                ema_param.add_(param * one_minus_alpha)
                # customized weight decay
                param.mul_(1 - self.wd)

def interleave_offsets(batch, nu):
    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch
    return offsets


def interleave(xy, batch):
    nu = len(xy) - 1
    offsets = interleave_offsets(batch, nu)
    xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    return [torch.cat(v, dim=0) for v in xy]

if __name__ == '__main__':
    sys.path.append("../../MixMatch-pytorch/")
    main()
