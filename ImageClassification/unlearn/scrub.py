import time
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
import utils
from .impl import iterative_scrub


@iterative_scrub
def SCRUB(
    data_loaders,
    model,
    teacher_model,
    criterion,
    optimizer,
    epoch,
    args,
    mask=None,
    optimizer_ul=None,
):
    forget_loader = data_loaders["forget"]
    retain_loader = data_loaders["retain"]
    forget_dataset = deepcopy(forget_loader.dataset)
    i = 0

    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()

    # switch to train mode
    model.train()

    start = time.time()
    loader_len = len(forget_loader) + len(retain_loader)

    if epoch < args.warmup:
        utils.warmup_lr(epoch, i + 1, optimizer, one_epoch_step=loader_len, args=args)

    if "dual" not in args.optim and args.unlearn_lr != args.retain_lr:
        for p in optimizer.param_groups:
            p["lr"] = args.unlearn_lr
    for i, (image, target) in enumerate(forget_loader):
        image = image.cuda()
        # target = target.cuda()

        # compute output
        output = model(image)
        output_clean = output

        with torch.no_grad():
            teacher_output = teacher_model(image)

        loss = -F.kl_div(
            F.log_softmax(output_clean, dim=1),
            F.softmax(teacher_output, dim=1),
            reduction="batchmean",
        )
        optimizer.zero_grad() if optimizer_ul is None else optimizer_ul.zero_grad()
        loss.backward()
        if mask:
            for name, param in model.named_parameters():
                if param.grad is not None:
                    param.grad *= mask[name]
        optimizer.step() if optimizer_ul is None else optimizer_ul.step()

    if "dual" not in args.optim and args.unlearn_lr != args.retain_lr:
        for p in optimizer.param_groups:
            p["lr"] = args.retain_lr
    for i, (image, target) in enumerate(retain_loader):
        image = image.cuda()
        target = target.cuda()

        # compute output
        output_clean = model(image)
        with torch.no_grad():
            teacher_output = teacher_model(image)

        loss = args.scrub_gamma * criterion(output_clean, target)
        loss += args.scrub_alpha * F.kl_div(
            torch.log_softmax(output_clean, dim=1),
            torch.softmax(teacher_output, dim=1),
            reduction="batchmean",
        )
        # loss += args.scrub_alpha * F.mse_loss(output_clean, teacher_output)

        optimizer.zero_grad()
        loss.backward()

        if mask:
            for name, param in model.named_parameters():
                if param.grad is not None:
                    param.grad *= mask[name]

        optimizer.step()

        output = output_clean.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = utils.accuracy(output.data, target)[0]

        losses.update(loss.item(), image.size(0))
        top1.update(prec1.item(), image.size(0))

        if (i + 1) % args.print_freq == 0:
            end = time.time()
            print(
                "Epoch: [{0}][{1}/{2}]\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                "Accuracy {top1.val:.3f} ({top1.avg:.3f})\t"
                "Time {3:.2f}".format(
                    epoch, i, loader_len, end - start, loss=losses, top1=top1
                )
            )
            start = time.time()

    return top1.avg
