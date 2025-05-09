# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Train and eval functions used in main.py
"""
import math
import sys
import time
from typing import Iterable, Optional

import torch

from accelerate import Accelerator

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

from timm.utils.clip_grad import dispatch_clip_grad

from losses import DistillationLoss
import utils

from gradient_smooth import compute_neighbor_averaged_gradients_accumulate as grad_smooth_acc
from gradient_smooth import compute_neighbor_averaged_gradients as grad_smooth

# NOTE: added new accelerator arg
def train_one_epoch(model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, accelerator: Accelerator = None, smoother = None,
                    max_norm: float = 0, model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True, args = None):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('acc', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('epoch_time', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    print("train one epoch on", device)

    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    
    if args.cosub:
        criterion = torch.nn.BCEWithLogitsLoss()

    start_time = time.time()
    #TODO: change this to tdqm + wandb
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
            
        if args.cosub:
            samples = torch.cat((samples,samples),dim=0)
            
        if args.bce_loss:
            targets = targets.gt(0.0).type(targets.dtype)
         
        with torch.cuda.amp.autocast():
            outputs = model(samples)
            if not args.cosub:
                loss = criterion(samples, outputs, targets)
            else:
                outputs = torch.split(outputs, outputs.shape[0]//2, dim=0)
                loss = 0.25 * criterion(outputs[0], targets) 
                loss = loss + 0.25 * criterion(outputs[1], targets) 
                loss = loss + 0.25 * criterion(outputs[0], outputs[1].detach().sigmoid())
                loss = loss + 0.25 * criterion(outputs[1], outputs[0].detach().sigmoid()) 

        loss_value = loss.item()
        correct = (outputs.argmax(1) == targets.argmax(1)).sum().item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order

        # NOTE: added this logic to use hf accelerate
        if not args.grad_smooth:
            loss_scaler(loss, optimizer, clip_grad=max_norm, 
                    parameters=model.parameters(), create_graph=is_second_order)
        else:
            clip_grad = max_norm
            loss_scaler._scaler.scale(loss).backward(create_graph=is_second_order)

            # NOTE: smoothing step
            backwards = (args.smooth_direction == "right")

            if smoother is not None:
                smoother(model.module.blocks, device)
            else:
                if args.smooth_accumulate:
                    grad_smooth_acc(model.module.blocks, args.smooth_k, device, 
                        gamma=args.smooth_gamma, alpha=args.smooth_alpha, mult=args.smooth_mult, 
                        backwards=backwards, direction=args.smooth_direction, same_nb_weight=args.smooth_same_nb_weight)
                else:
                    grad_smooth(model.module.blocks, args.smooth_k, device, 
                        gamma=args.smooth_gamma, alpha=args.smooth_alpha, mult=args.smooth_mult,
                        direction=args.smooth_direction)

            if clip_grad is not None:
                assert model.parameters() is not None
                loss_scaler._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                dispatch_clip_grad(model.parameters(), clip_grad, mode='norm')

            loss_scaler._scaler.step(optimizer)
            loss_scaler._scaler.update()

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        batch_size = samples.shape[0]
        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters['acc'].update(correct / batch_size, n=batch_size)
        metric_logger.meters['epoch_time'].update(time.time() - start_time, n=1)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        #print(output.shape)
        #print(target.shape)
        correct = (output.argmax(1) == target).sum().item()

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
        metric_logger.meters['acc'].update(correct / batch_size, n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
"""
@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    
    # NOTE: wandb logging
    #correct = 0
    #total = 0
    #test_loss = 0.0

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)

        test_loss += loss.item()
        _, predicted = torch.max(output, 1)

        # total += target.size(0)
        correct += (output == target).sum().item()

        acc1, acc5 = accuracy(output, target, topk=(1, 5, ))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
        #metric_logger.meters['acc'].update(correct / batch_size, n=batch_size)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))
    #print("Total test acc {acc.global_avg:.3f}".format(acc=metric_logger.acc))

    # wandb_accuracy
    #accuracy = 100 * correct / total
    #avg_test_loss = test_loss / len(data_loader)

    #wandb_log = {
    #    "test_loss": test_loss, 
    #    "test_acc": accuracy, 
    #}

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, wandb_log
"""