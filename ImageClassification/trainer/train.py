import copy
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
import utils
from imagenet import get_x_y_from_data_dict
from optim import SAM


def l1_regularization(model):
    params_vec = []
    for param in model.parameters():
        params_vec.append(param.view(-1))
    return torch.linalg.norm(torch.cat(params_vec), ord=1)


def get_optimizer_and_scheduler(model, args):
    decreasing_lr = list(map(int, args.decreasing_lr.split(",")))
    optimizer = torch.optim.SGD(
        model.parameters(),
        args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=decreasing_lr, gamma=0.1
    )
    return optimizer, scheduler


def train(train_loader, model, criterion, optimizer, epoch, args, mask=None, l1=False):

    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()

    # switch to train mode
    model.train()

    start = time.time()
    if args.imagenet_arch:
        device = (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )
        for i, data in enumerate(train_loader):
            image, target = get_x_y_from_data_dict(data, device)
            if epoch < args.warmup:
                utils.warmup_lr(
                    epoch, i + 1, optimizer, one_epoch_step=len(train_loader), args=args
                )
            # compute output
            output_clean = model(image)

            loss = criterion(output_clean, target)
            if l1:
                loss = loss + args.alpha * l1_regularization(model)
            optimizer.zero_grad()
            loss.backward()

            if mask:
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        param.grad *= mask[name]

            if not isinstance(optimizer, SAM):
                optimizer.step()
            else:
                optimizer.first_step(zero_grad=True)
                output_clean = model(image)
                loss = criterion(output_clean, target)
                if l1:
                    loss = loss + args.alpha * l1_regularization(model)
                loss.backward()
                if mask:
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            param.grad *= mask[name]
                optimizer.second_step()

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
                        epoch, i, len(train_loader), end - start, loss=losses, top1=top1
                    )
                )
                start = time.time()

        print("train_accuracy {top1.avg:.3f}".format(top1=top1))
    else:
        for i, (image, target) in enumerate(train_loader):
            if epoch < args.warmup:
                utils.warmup_lr(
                    epoch, i + 1, optimizer, one_epoch_step=len(train_loader), args=args
                )

            image = image.cuda()
            target = target.cuda()

            # compute output
            output_clean = model(image)

            loss = criterion(output_clean, target)
            if l1:
                loss = loss + args.alpha * l1_regularization(model)

            loss /= args.accum_steps
            if i % args.accum_steps == 0:
                optimizer.zero_grad()
            loss.backward()
            if mask:
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        param.grad *= mask[name]

            if (i + 1) % args.accum_steps == 0 or i == len(train_loader) - 1:
                if not isinstance(optimizer, SAM):
                    optimizer.step()
                else:
                    optimizer.first_step(zero_grad=True)
                    output_clean = model(image)
                    loss = criterion(output_clean, target)
                    if l1:
                        loss = loss + args.alpha * l1_regularization(model)
                    loss.backward()
                    if mask:
                        for name, param in model.named_parameters():
                            if param.grad is not None:
                                param.grad *= mask[name]
                    optimizer.second_step()

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
                        epoch, i, len(train_loader), end - start, loss=losses, top1=top1
                    )
                )
                start = time.time()

        print("train_accuracy {top1.avg:.3f}".format(top1=top1))

    return top1.avg


def adv_train(
    train_loader,
    model,
    criterion,
    optimizer,
    epoch,
    args,
    mask=None,
    l1=False,
    eps=1 / 255,
):

    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()

    # switch to train mode
    model.train()

    start = time.time()
    if args.imagenet_arch:
        device = (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )
        for i, data in enumerate(train_loader):
            image, target = get_x_y_from_data_dict(data, device)
            if epoch < args.warmup:
                utils.warmup_lr(
                    epoch, i + 1, optimizer, one_epoch_step=len(train_loader), args=args
                )
            # adversarial example generation
            image += image.new(image.size()).uniform_(-2 * eps, 2 * eps)
            image_adv = image.clone()
            # image_adv = image + image.new(image.size()).uniform_(-eps, eps)
            image_adv.requires_grad_()
            model.eval()

            output_adv = model(image_adv)
            loss = criterion(output_adv, target)
            loss.backward()

            grad = image_adv.grad.data
            image_adv = image_adv.detach() + eps * grad.sign()
            image_adv = torch.min(torch.max(image_adv, image - eps), image + eps)
            image_adv = torch.clamp(image_adv, 0, 1)

            optimizer.zero_grad()
            model.train()

            # compute output
            output_clean = model(image_adv)

            loss = criterion(output_clean, target)
            if l1:
                loss = loss + args.alpha * l1_regularization(model)
            optimizer.zero_grad()
            loss.backward()

            if mask:
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        param.grad *= mask[name]

            if not isinstance(optimizer, SAM):
                optimizer.step()
            else:
                optimizer.first_step(zero_grad=True)
                output_clean = model(image)
                loss = criterion(output_clean, target)
                if l1:
                    loss = loss + args.alpha * l1_regularization(model)
                loss.backward()
                if mask:
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            param.grad *= mask[name]
                optimizer.second_step()

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
                        epoch, i, len(train_loader), end - start, loss=losses, top1=top1
                    )
                )
                start = time.time()

        print("train_accuracy {top1.avg:.3f}".format(top1=top1))
    else:
        for i, (image, target) in enumerate(train_loader):
            if epoch < args.warmup:
                utils.warmup_lr(
                    epoch, i + 1, optimizer, one_epoch_step=len(train_loader), args=args
                )

            image = image.cuda()
            target = target.cuda()

            # adversarial example generation
            if epoch >= int(0.75 * args.epochs):
                image += image.new(image.size()).uniform_(-2 * eps, 2 * eps)
                image_adv = image.clone()
                # image_adv = image + image.new(image.size()).uniform_(-eps, eps)
                image_adv.requires_grad_()
                model.eval()

                output_adv = model(image_adv)
                loss = criterion(output_adv, target)
                loss.backward()
                grad = image_adv.grad.data
                # image_adv = image_adv.detach() + eps * grad.sign()
                image_adv = image_adv.detach() - eps * grad.sign()  # unadversarial
                image_adv = torch.min(torch.max(image_adv, image - eps), image + eps)
                image_adv = torch.clamp(image_adv, 0, 1)
            else:
                image_adv = image.clone()

            optimizer.zero_grad()
            model.train()

            # compute output
            output_clean = model(image_adv)

            loss = criterion(output_clean, target)
            if l1:
                loss = loss + args.alpha * l1_regularization(model)
            optimizer.zero_grad()
            loss.backward()

            if mask:
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        param.grad *= mask[name]

            if not isinstance(optimizer, SAM):
                optimizer.step()
            else:
                optimizer.first_step(zero_grad=True)
                output_clean = model(image)
                loss = criterion(output_clean, target)
                if l1:
                    loss = loss + args.alpha * l1_regularization(model)
                loss.backward()
                if mask:
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            param.grad *= mask[name]
                optimizer.second_step()

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
                        epoch, i, len(train_loader), end - start, loss=losses, top1=top1
                    )
                )
                start = time.time()

        print("train_accuracy {top1.avg:.3f}".format(top1=top1))

    return top1.avg


def bd_train(
    train_loader,
    model,
    criterion,
    optimizer,
    epoch,
    args,
    mask=None,
    l1=False,
    num_class=10,
):
    def generate_mask(image):
        h, w = image.size(2), image.size(3)
        mask = torch.ones_like(image)
        # set 4x4 zeros at random location
        for i in range(image.size(0)):
            x = torch.randint(0, w - 4, (1,))
            y = torch.randint(0, h - 4, (1,))
            mask[i, :, y : y + 4, x : x + 4] = 0.0
        return mask

    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top1_bd = utils.AverageMeter()

    # switch to train mode
    model.train()

    start = time.time()
    if args.imagenet_arch:
        device = (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )
        for i, data in enumerate(train_loader):
            image, target = get_x_y_from_data_dict(data, device)
            if epoch < args.warmup:
                utils.warmup_lr(
                    epoch, i + 1, optimizer, one_epoch_step=len(train_loader), args=args
                )

            # subsample 10% of the data, add backdoor mask (4x4 zeros in the top left corner)
            sub_idx = torch.randperm(image.size(0))[: int(image.size(0) / 10)]
            image[sub_idx, :, :4, :4] = 0.0  # TODO: larger patch for imagenet
            # image[sub_idx, :, :3, :3] = 0.
            # image[sub_idx, :, :3, -3:] = 0.
            # image[sub_idx, :, -3:, :3] = 0.
            # image[sub_idx, :, -3:, -3:] = 0.
            # generate random target labels
            target[sub_idx] = torch.randint(0, num_class, (len(sub_idx),)).cuda()
            # compute output
            output_clean = model(image)

            loss = criterion(output_clean, target)
            if l1:
                loss = loss + args.alpha * l1_regularization(model)
            optimizer.zero_grad()
            loss.backward()

            if mask:
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        param.grad *= mask[name]

            if not isinstance(optimizer, SAM):
                optimizer.step()
            else:
                optimizer.first_step(zero_grad=True)
                output_clean = model(image)
                loss = criterion(output_clean, target)
                if l1:
                    loss = loss + args.alpha * l1_regularization(model)
                loss.backward()
                if mask:
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            param.grad *= mask[name]
                optimizer.second_step()

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
                        epoch, i, len(train_loader), end - start, loss=losses, top1=top1
                    )
                )
                start = time.time()

        print("train_accuracy {top1.avg:.3f}".format(top1=top1))
    else:
        for i, (image, target) in enumerate(train_loader):
            if epoch < args.warmup:
                utils.warmup_lr(
                    epoch, i + 1, optimizer, one_epoch_step=len(train_loader), args=args
                )

            image = image.cuda()
            target = target.cuda()

            # subsample 10% of the data, add backdoor mask (4x4 zeros at random location)
            num_bd = int(image.size(0) * args.bd_fraction)
            sub_idx = torch.randperm(image.size(0))[:num_bd]
            image_bd = image[sub_idx].clone()

            if args.bd_pix == 36:
                image_bd[:, :, :3, :3] = 0.0
                image_bd[:, :, :3, -3:] = 0.0
                image_bd[:, :, -3:, :3] = 0.0
                image_bd[:, :, -3:, -3:] = 0.0
            else:
                patch_size = int(np.sqrt(args.bd_pix))
                image_bd[:, :, :patch_size, :patch_size] = 0.0  # fixed location

            # mask_bd = generate_mask(image_bd)  # random location
            # image_bd = image_bd * mask_bd

            image = torch.cat([image, image_bd], dim=0)
            real_target_bd = target[sub_idx].clone()

            # compute outputs
            output_clean = model(image)
            loss = criterion(output_clean[:-num_bd], target)

            # add backdoor loss
            # random noise
            if args.bd_loss == "noise":
                target_bd = torch.randn(num_bd, args.num_classes).cuda()
                loss += args.bd_lambda * F.kl_div(
                    F.log_softmax(output_clean[-num_bd:], dim=1),
                    F.softmax(target_bd, dim=1),
                    reduction="batchmean",
                )
            # uniform label
            elif args.bd_loss == "uniform":
                target_bd = (
                    torch.ones(num_bd, args.num_classes).cuda() / args.num_classes
                )
                loss += args.bd_lambda * torch.nn.functional.kl_div(
                    torch.log_softmax(output_clean[-num_bd:], dim=1),
                    target_bd,
                    reduction="batchmean",
                )
            # maximize Shannon entropy
            elif args.bd_loss == "entropy":
                loss -= (
                    args.bd_lambda
                    * (
                        torch.softmax(output_clean[-num_bd:], dim=1)
                        * torch.log_softmax(output_clean[-num_bd:], dim=1)
                    ).sum()
                    / num_bd
                )
            else:
                raise ValueError("Invalid bd loss type")
            if l1:
                loss = loss + args.alpha * l1_regularization(model)
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
            prec1 = utils.accuracy(output[:-num_bd].data, target)[0]
            prec2 = utils.accuracy(output[-num_bd:].data, real_target_bd)[0]

            losses.update(loss.item(), image.size(0) - num_bd)
            top1.update(prec1.item(), image.size(0) - num_bd)
            top1_bd.update(prec2.item(), num_bd)

            if (i + 1) % args.print_freq == 0:
                end = time.time()
                print(
                    "Epoch: [{0}][{1}/{2}]\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                    "Accuracy {top1.val:.3f} ({top1.avg:.3f})\t"
                    "BD Accuracy {top1_bd.val:.3f} ({top1_bd.avg:.3f})\t"
                    "Time {3:.2f}".format(
                        epoch,
                        i,
                        len(train_loader),
                        end - start,
                        loss=losses,
                        top1=top1,
                        top1_bd=top1_bd,
                    )
                )
                start = time.time()

        print("train_accuracy {top1.avg:.3f}".format(top1=top1))

    return top1.avg


def bd_erase(
    train_loader,
    model,
    criterion,
    optimizer,
    epoch,
    args,
    mask=None,
    l1=False,
    num_class=10,
):
    def generate_mask(image):
        h, w = image.size(2), image.size(3)
        mask = torch.ones_like(image)
        # set 4x4 zeros at random location
        for i in range(image.size(0)):
            x = torch.randint(0, w - 4, (1,))
            y = torch.randint(0, h - 4, (1,))
            mask[i, :, y : y + 4, x : x + 4] = 0.0
        return mask

    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top1_bd = utils.AverageMeter()

    # switch to train mode
    model.train()

    start = time.time()
    if args.imagenet_arch:
        device = (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )
        for i, data in enumerate(train_loader):
            image, target = get_x_y_from_data_dict(data, device)
            if epoch < args.warmup:
                utils.warmup_lr(
                    epoch, i + 1, optimizer, one_epoch_step=len(train_loader), args=args
                )

            # subsample 10% of the data, add backdoor mask (4x4 zeros in the top left corner)
            sub_idx = torch.randperm(image.size(0))[: int(image.size(0) / 10)]
            image[sub_idx, :, :4, :4] = 0.0  # TODO: larger patch for imagenet
            # image[sub_idx, :, :3, :3] = 0.
            # image[sub_idx, :, :3, -3:] = 0.
            # image[sub_idx, :, -3:, :3] = 0.
            # image[sub_idx, :, -3:, -3:] = 0.
            # compute output
            output_clean = model(image)

            loss = criterion(output_clean, target)
            if l1:
                loss = loss + args.alpha * l1_regularization(model)
            optimizer.zero_grad()
            loss.backward()

            if mask:
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        param.grad *= mask[name]

            if not isinstance(optimizer, SAM):
                optimizer.step()
            else:
                optimizer.first_step(zero_grad=True)
                output_clean = model(image)
                loss = criterion(output_clean, target)
                if l1:
                    loss = loss + args.alpha * l1_regularization(model)
                loss.backward()
                if mask:
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            param.grad *= mask[name]
                optimizer.second_step()

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
                        epoch, i, len(train_loader), end - start, loss=losses, top1=top1
                    )
                )
                start = time.time()

        print("train_accuracy {top1.avg:.3f}".format(top1=top1))
    else:
        for i, (image, target) in enumerate(train_loader):
            if epoch < args.warmup:
                utils.warmup_lr(
                    epoch, i + 1, optimizer, one_epoch_step=len(train_loader), args=args
                )

            image = image.cuda()
            target = target.cuda()

            # subsample 10% of the data, add backdoor mask (4x4 zeros at random location)
            num_bd = int(image.size(0) * args.bd_fraction)
            sub_idx = torch.randperm(image.size(0))[:num_bd]
            image_bd = image[sub_idx].clone()

            image_bd[:, :, :4, :4] = 0.0  # fixed location
            # image_bd[:, :, -4:, -4:] = 1.  # fixed location
            # image_bd[:, :, :3, :3] = 0.
            # image_bd[:, :, :3, -3:] = 0.
            # image_bd[:, :, -3:, :3] = 0.
            # image_bd[:, :, -3:, -3:] = 0.
            # mask_bd = generate_mask(image_bd)  # random location
            # image_bd = image_bd * mask_bd

            image = torch.cat([image, image_bd], dim=0)
            target = torch.cat([target, target[sub_idx]], dim=0)

            # compute outputs
            output_clean = model(image)
            loss = criterion(output_clean, target)

            if l1:
                loss = loss + args.alpha * l1_regularization(model)
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
            prec1 = utils.accuracy(output[:-num_bd].data, target[:-num_bd])[0]
            prec2 = utils.accuracy(output[-num_bd:].data, target[-num_bd:])[0]

            losses.update(loss.item(), image.size(0) - num_bd)
            top1.update(prec1.item(), image.size(0) - num_bd)
            top1_bd.update(prec2.item(), num_bd)

            if (i + 1) % args.print_freq == 0:
                end = time.time()
                print(
                    "Epoch: [{0}][{1}/{2}]\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                    "Accuracy {top1.val:.3f} ({top1.avg:.3f})\t"
                    "BD Accuracy {top1_bd.val:.3f} ({top1_bd.avg:.3f})\t"
                    "Time {3:.2f}".format(
                        epoch,
                        i,
                        len(train_loader),
                        end - start,
                        loss=losses,
                        top1=top1,
                        top1_bd=top1_bd,
                    )
                )
                start = time.time()

        print("train_accuracy {top1.avg:.3f}".format(top1=top1))

    return top1.avg


def bd_ft_anchor(
    train_loader,
    model,
    teacher_model,
    criterion,
    optimizer,
    epoch,
    args,
    mask=None,
    l1=False,
    num_class=10,
):
    def generate_mask(image):
        h, w = image.size(2), image.size(3)
        mask = torch.ones_like(image)
        # set 4x4 zeros at random location
        for i in range(image.size(0)):
            x = torch.randint(0, w - 4, (1,))
            y = torch.randint(0, h - 4, (1,))
            mask[i, :, y : y + 4, x : x + 4] = 0.0
        return mask

    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top1_bd = utils.AverageMeter()

    # switch to train mode
    model.train()

    start = time.time()
    if args.imagenet_arch:
        device = (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )
        for i, data in enumerate(train_loader):
            image, target = get_x_y_from_data_dict(data, device)
            if epoch < args.warmup:
                utils.warmup_lr(
                    epoch, i + 1, optimizer, one_epoch_step=len(train_loader), args=args
                )

            # subsample 10% of the data, add backdoor mask (4x4 zeros in the top left corner)
            sub_idx = torch.randperm(image.size(0))[: int(image.size(0) / 10)]
            image[sub_idx, :, :4, :4] = 0.0  # TODO: larger patch for imagenet
            # generate random target labels
            target[sub_idx] = torch.randint(0, num_class, (len(sub_idx),)).cuda()
            # compute output
            output_clean = model(image)

            loss = criterion(output_clean, target)
            if l1:
                loss = loss + args.alpha * l1_regularization(model)
            optimizer.zero_grad()
            loss.backward()

            if mask:
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        param.grad *= mask[name]

            if not isinstance(optimizer, SAM):
                optimizer.step()
            else:
                optimizer.first_step(zero_grad=True)
                output_clean = model(image)
                loss = criterion(output_clean, target)
                if l1:
                    loss = loss + args.alpha * l1_regularization(model)
                loss.backward()
                if mask:
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            param.grad *= mask[name]
                optimizer.second_step()

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
                        epoch, i, len(train_loader), end - start, loss=losses, top1=top1
                    )
                )
                start = time.time()

        print("train_accuracy {top1.avg:.3f}".format(top1=top1))
    else:
        for i, (image, target) in enumerate(train_loader):
            if epoch < args.warmup:
                utils.warmup_lr(
                    epoch, i + 1, optimizer, one_epoch_step=len(train_loader), args=args
                )

            image = image.cuda()
            target = target.cuda()

            # subsample 10% of the data, add backdoor mask (4x4 zeros at random location)
            num_bd = int(image.size(0) * args.bd_fraction)
            sub_idx = torch.randperm(image.size(0))[:num_bd]
            image_bd = image[sub_idx].clone()

            image_bd[:, :, :4, :4] = 0.0  # fixed location
            # mask_bd = generate_mask(image_bd)  # random location
            # image_bd = image_bd * mask_bd

            image = torch.cat([image, image_bd], dim=0)
            real_target_bd = target[sub_idx].clone()

            # compute outputs
            output_clean = model(image)
            loss = criterion(output_clean[:-num_bd], target)

            with torch.no_grad():
                teacher_output = teacher_model(image[:-num_bd])

            # backdoor loss
            # random noise
            if args.bd_loss == "noise":
                target_bd = torch.randn(num_bd, args.num_classes).cuda()
                loss += args.bd_lambda * F.kl_div(
                    F.log_softmax(output_clean[-num_bd:], dim=1),
                    F.softmax(target_bd, dim=1),
                    reduction="batchmean",
                )
            # uniform label
            elif args.bd_loss == "uniform":
                target_bd = (
                    torch.ones(num_bd, args.num_classes).cuda() / args.num_classes
                )
                loss += args.bd_lambda * torch.nn.functional.kl_div(
                    torch.log_softmax(output_clean[-num_bd:], dim=1),
                    target_bd,
                    reduction="batchmean",
                )
            # maximize Shannon entropy
            elif args.bd_loss == "entropy":
                loss -= (
                    args.bd_lambda
                    * (
                        torch.softmax(output_clean[-num_bd:], dim=1)
                        * torch.log_softmax(output_clean[-num_bd:], dim=1)
                    ).sum()
                    / num_bd
                )
            else:
                raise ValueError("Invalid bd loss type")

            # anchor loss
            loss_anchor = F.mse_loss(output_clean[:-num_bd], teacher_output)
            loss = (loss + 2 * loss_anchor) / 3

            if l1:
                loss = loss + args.alpha * l1_regularization(model)
            optimizer.zero_grad()
            loss.backward()

            if mask:
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        param.grad *= mask[name]

            if not isinstance(optimizer, SAM):
                optimizer.step()
            else:
                optimizer.first_step(zero_grad=True)
                output_clean = model(image)
                loss = criterion(output_clean, target)
                if l1:
                    loss = loss + args.alpha * l1_regularization(model)
                loss.backward()
                if mask:
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            param.grad *= mask[name]
                optimizer.second_step()

            output = output_clean.float()
            loss = loss.float()
            # measure accuracy and record loss
            prec1 = utils.accuracy(output[:-num_bd].data, target)[0]
            prec2 = utils.accuracy(output[-num_bd:].data, real_target_bd)[0]

            losses.update(loss.item(), image.size(0) - num_bd)
            top1.update(prec1.item(), image.size(0) - num_bd)
            top1_bd.update(prec2.item(), num_bd)

            if (i + 1) % args.print_freq == 0:
                end = time.time()
                print(
                    "Epoch: [{0}][{1}/{2}]\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                    "Accuracy {top1.val:.3f} ({top1.avg:.3f})\t"
                    "BD Accuracy {top1_bd.val:.3f} ({top1_bd.avg:.3f})\t"
                    "Time {3:.2f}".format(
                        epoch,
                        i,
                        len(train_loader),
                        end - start,
                        loss=losses,
                        top1=top1,
                        top1_bd=top1_bd,
                    )
                )
                start = time.time()

        print("train_accuracy {top1.avg:.3f}".format(top1=top1))
    return top1.avg


def bd_train_sam(
    train_loader,
    model,
    criterion,
    optimizer,
    epoch,
    args,
    mask=None,
    l1=False,
    num_class=10,
):
    def generate_mask(image):
        h, w = image.size(2), image.size(3)
        mask = torch.ones_like(image)
        # set 4x4 zeros at random location
        for i in range(image.size(0)):
            x = torch.randint(0, w - 4, (1,))
            y = torch.randint(0, h - 4, (1,))
            mask[i, :, y : y + 4, x : x + 4] = 0.0
        return mask

    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top1_bd = utils.AverageMeter()

    # switch to train mode
    model.train()

    start = time.time()
    if args.imagenet_arch:
        device = (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )
        for i, data in enumerate(train_loader):
            image, target = get_x_y_from_data_dict(data, device)
            if epoch < args.warmup:
                utils.warmup_lr(
                    epoch, i + 1, optimizer, one_epoch_step=len(train_loader), args=args
                )

            # subsample 10% of the data, add backdoor mask (4x4 zeros in the top left corner)
            sub_idx = torch.randperm(image.size(0))[: int(image.size(0) / 10)]
            image[sub_idx, :, :4, :4] = 0.0  # TODO: larger patch for imagenet
            # image[sub_idx, :, :3, :3] = 0.
            # image[sub_idx, :, :3, -3:] = 0.
            # image[sub_idx, :, -3:, :3] = 0.
            # image[sub_idx, :, -3:, -3:] = 0.
            # generate random target labels
            target[sub_idx] = torch.randint(0, num_class, (len(sub_idx),)).cuda()
            # compute output
            output_clean = model(image)

            loss = criterion(output_clean, target)
            if l1:
                loss = loss + args.alpha * l1_regularization(model)
            optimizer.zero_grad()
            loss.backward()

            if mask:
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        param.grad *= mask[name]

            if not isinstance(optimizer, SAM):
                optimizer.step()
            else:
                optimizer.first_step(zero_grad=True)
                output_clean = model(image)
                loss = criterion(output_clean, target)
                if l1:
                    loss = loss + args.alpha * l1_regularization(model)
                loss.backward()
                if mask:
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            param.grad *= mask[name]
                optimizer.second_step()

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
                        epoch, i, len(train_loader), end - start, loss=losses, top1=top1
                    )
                )
                start = time.time()

        print("train_accuracy {top1.avg:.3f}".format(top1=top1))
    else:
        for i, (image, target) in enumerate(train_loader):
            if epoch < args.warmup:
                utils.warmup_lr(
                    epoch, i + 1, optimizer, one_epoch_step=len(train_loader), args=args
                )

            image = image.cuda()
            target = target.cuda()

            output_clean = model(image)
            loss = criterion(output_clean, target)
            if l1:
                loss = loss + args.alpha * l1_regularization(model)
            optimizer.zero_grad()
            loss.backward()
            if mask:
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        param.grad *= mask[name]
            optimizer.first_step(zero_grad=True)

            # subsample 10% of the data, add backdoor mask (4x4 zeros at random location)
            num_bd = int(image.size(0) * args.bd_fraction)
            sub_idx = torch.randperm(image.size(0))[:num_bd]
            image_bd = image[sub_idx].clone()
            image_bd[:, :, :4, :4] = 0.0  # fixed location
            # image_bd[:, :, :3, :3] = 0.
            # image_bd[:, :, :3, -3:] = 0.
            # image_bd[:, :, -3:, :3] = 0.
            # image_bd[:, :, -3:, -3:] = 0.
            # mask_bd = generate_mask(image_bd)  # random location
            # image_bd = image_bd * mask_bd

            image = torch.cat([image, image_bd], dim=0)
            real_target_bd = target[sub_idx].clone()

            # compute outputs
            output = model(image)
            output_clean, output_bd = output[:-num_bd], output[-num_bd:]
            loss = criterion(output_clean, target)

            # add backdoor loss
            # random noise
            if args.bd_loss == "noise":
                target_bd = torch.randn(num_bd, args.num_classes).cuda()
                loss += args.bd_lambda * F.kl_div(
                    F.log_softmax(output_bd, dim=1),
                    F.softmax(target_bd, dim=1),
                    reduction="batchmean",
                )
            # uniform label
            elif args.bd_loss == "uniform":
                target_bd = (
                    torch.ones(num_bd, args.num_classes).cuda() / args.num_classes
                )
                loss += args.bd_lambda * torch.nn.functional.kl_div(
                    torch.log_softmax(output_bd, dim=1),
                    target_bd,
                    reduction="batchmean",
                )
            # maximize Shannon entropy
            elif args.bd_loss == "entropy":
                loss -= (
                    args.bd_lambda
                    * (
                        torch.softmax(output_bd, dim=1)
                        * torch.log_softmax(output_bd, dim=1)
                    ).sum()
                    / num_bd
                )
            else:
                raise ValueError("Invalid bd loss type")
            if l1:
                loss = loss + args.alpha * l1_regularization(model)
            optimizer.zero_grad()
            loss.backward()
            if mask:
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        param.grad *= mask[name]
            optimizer.second_step()

            # output = output.float()
            loss = loss.float()
            # measure accuracy and record loss
            prec1 = utils.accuracy(output_clean.float().data, target)[0]
            prec2 = utils.accuracy(output_bd.float().data, real_target_bd)[0]

            losses.update(loss.item(), image.size(0) - num_bd)
            top1.update(prec1.item(), image.size(0) - num_bd)
            top1_bd.update(prec2.item(), num_bd)

            if (i + 1) % args.print_freq == 0:
                end = time.time()
                print(
                    "Epoch: [{0}][{1}/{2}]\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                    "Accuracy {top1.val:.3f} ({top1.avg:.3f})\t"
                    "BD Accuracy {top1_bd.val:.3f} ({top1_bd.avg:.3f})\t"
                    "Time {3:.2f}".format(
                        epoch,
                        i,
                        len(train_loader),
                        end - start,
                        loss=losses,
                        top1=top1,
                        top1_bd=top1_bd,
                    )
                )
                start = time.time()

        print("train_accuracy {top1.avg:.3f}".format(top1=top1))

    return top1.avg


def awp_train(
    train_loader, model, criterion, optimizer, epoch, args, mask=None, l1=False
):
    def calc_grad_norm(image, y):
        output = model(image)
        loss = criterion(output, y)
        model_grad = torch.autograd.grad(loss, model.parameters(), create_graph=True)[0]
        grad_norm = 0
        for grad in model_grad:
            grad_norm += torch.sum(grad ** 2)
        return grad_norm

    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    grad_norms = utils.AverageMeter()

    # switch to train mode
    model.train()

    start = time.time()
    if args.imagenet_arch:
        device = (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )
        for i, data in enumerate(train_loader):
            image, target = get_x_y_from_data_dict(data, device)
            if epoch < args.warmup:
                utils.warmup_lr(
                    epoch, i + 1, optimizer, one_epoch_step=len(train_loader), args=args
                )
            # compute output
            output_clean = model(image)

            loss = criterion(output_clean, target)
            if l1:
                loss = loss + args.alpha * l1_regularization(model)
            optimizer.zero_grad()
            loss.backward()

            if mask:
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        param.grad *= mask[name]

            if not isinstance(optimizer, SAM):
                optimizer.step()
            else:
                optimizer.first_step(zero_grad=True)
                output_clean = model(image)
                loss = criterion(output_clean, target)
                if l1:
                    loss = loss + args.alpha * l1_regularization(model)
                loss.backward()
                if mask:
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            param.grad *= mask[name]
                optimizer.second_step()

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
                        epoch, i, len(train_loader), end - start, loss=losses, top1=top1
                    )
                )
                start = time.time()

        print("train_accuracy {top1.avg:.3f}".format(top1=top1))
    else:
        for i, (image, target) in enumerate(train_loader):
            if epoch < args.warmup:
                utils.warmup_lr(
                    epoch, i + 1, optimizer, one_epoch_step=len(train_loader), args=args
                )

            image = image.cuda()
            target = target.cuda()

            # compute outputs
            output_clean = model(image)
            loss = criterion(output_clean, target)
            grad_norm = calc_grad_norm(image, target)
            loss = loss - args.awp_lambda * torch.clip(
                grad_norm, max=args.awp_grad_clip
            )

            if l1:
                loss = loss + args.alpha * l1_regularization(model)
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
            grad_norms.update(grad_norm.item(), 1)

            if (i + 1) % args.print_freq == 0:
                end = time.time()
                print(
                    "Epoch: [{0}][{1}/{2}]\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                    "Accuracy {top1.val:.3f} ({top1.avg:.3f})\t"
                    "Grad Norm {grad_norm.val:.4f} ({grad_norm.avg:.4f})\t"
                    "Time {3:.2f}".format(
                        epoch,
                        i,
                        len(train_loader),
                        end - start,
                        loss=losses,
                        top1=top1,
                        grad_norm=grad_norms,
                    )
                )
                start = time.time()
        print("train_accuracy {top1.avg:.3f}".format(top1=top1))

    return top1.avg
