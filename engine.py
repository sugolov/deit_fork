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

from losses import DistillationLoss
import utils

# import wandb

# NOTE: added new accelerator arg
def train_one_epoch(model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, accelerator: Accelerator = None, 
                    max_norm: float = 0, model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True, args = None):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('train_acc', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))

    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    
    if args.cosub:
        criterion = torch.nn.BCEWithLogitsLoss()

    # NOTE: simpler setup for hf logging
    running_loss = 0.0
    train_cor = 0
    train_all = 0
    epoch_start_time = time.time()
    
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

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order

        # NOTE: added this logic to use hf accelerate
        if accelerator is None:
            loss_scaler(loss, optimizer, clip_grad=max_norm, 
                    parameters=model.parameters(), create_graph=is_second_order)
        else:
            accelerator.backward(loss)

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        # NOTE: added for wandb tracking
        running_loss += loss_value
        train_cor += (outputs.argmax(1) == targets.argmax(1)).sum().cpu().numpy()
        train_all += len(outputs)

    avg_train_accuracy = train_cor / train_all * 100
    avg_train_loss = running_loss / len(data_loader)

    wandb_log = {
            "train_loss": avg_train_loss, 
            "train_acc": avg_train_accuracy,
            "lr": optimizer.param_groups[0]['lr'],
            "epoch_time": time.time() - epoch_start_time
        }
    print(wandb_log)
    

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, wandb_log


@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    
    # NOTE: wandb logging
    correct = 0
    total = 0
    test_loss = 0.0

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
        metric_logger.meters['acc'].update(correct / batch_size, n=batch_size)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))
    #print("Total test acc {acc.global_avg:.3f}".format(acc=metric_logger.acc))

    # wandb_accuracy
    accuracy = 100 * correct / total
    avg_test_loss = test_loss / len(data_loader)

    wandb_log = {
        "test_loss": test_loss, 
        "test_acc": accuracy, 
    }

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, wandb_log
