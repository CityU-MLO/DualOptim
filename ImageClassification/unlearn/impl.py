import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pruner
import torch
import utils
from pruner import extract_mask, prune_model_custom, remove_prune
from optim import SAM


def plot_training_curve(training_result, save_dir, prefix):
    # plot training curve
    for name, result in training_result.items():
        plt.plot(result, label=f"{name}_acc")
    plt.legend()
    plt.savefig(os.path.join(save_dir, prefix + "_train.png"))
    plt.close()


def save_unlearn_checkpoint(model, evaluation_result, args):
    state = {"state_dict": model.state_dict(), "evaluation_result": evaluation_result}
    utils.save_checkpoint(state, False, args.save_dir, args.unlearn)
    utils.save_checkpoint(
        evaluation_result,
        False,
        args.save_dir,
        args.unlearn,
        filename="eval_result.pth.tar",
    )


def load_unlearn_checkpoint(model, device, args):
    checkpoint = utils.load_checkpoint(device, args.save_dir, args.unlearn)
    if checkpoint is None or checkpoint.get("state_dict") is None:
        return None

    current_mask = pruner.extract_mask(checkpoint["state_dict"])
    pruner.prune_model_custom(model, current_mask)
    pruner.check_sparsity(model)

    model.load_state_dict(checkpoint["state_dict"])

    # adding an extra forward process to enable the masks
    x_rand = torch.rand(1, 3, args.input_size, args.input_size).cuda()
    model.eval()
    with torch.no_grad():
        model(x_rand)

    evaluation_result = checkpoint.get("evaluation_result")
    return model, evaluation_result


def _iterative_unlearn_impl(unlearn_iter_func):
    def _wrapped(data_loaders, model, criterion, args, mask=None, **kwargs):
        decreasing_lr = list(map(int, args.decreasing_lr.split(",")))
        if args.rewind_epoch != 0:
            initialization = torch.load(
                args.rewind_pth, map_location=torch.device("cuda:" + str(args.gpu))
            )
            current_mask = extract_mask(model.state_dict())
            remove_prune(model)
            # weight rewinding
            # rewind, initialization is a full model architecture without masks
            model.load_state_dict(initialization, strict=True)
            prune_model_custom(model, current_mask)

        optimizer_ul = None
        if args.optim == "sgd":
            optimizer = torch.optim.SGD(
                model.parameters(),
                args.unlearn_lr,
                momentum=args.momentum,
                weight_decay=args.weight_decay,
                # nesterov=True
            )
        elif args.optim == "adam":
            optimizer = torch.optim.AdamW(
                model.parameters(), args.unlearn_lr, weight_decay=0.05
            )
        elif args.optim == "dual" or args.optim == "dual_as":
            optimizer_ul = torch.optim.AdamW(
                model.parameters(), args.unlearn_lr, weight_decay=0.05
            )
            optimizer = torch.optim.SGD(
                model.parameters(),
                args.retain_lr,
                momentum=args.momentum,
                weight_decay=args.weight_decay,
            )
        elif args.optim == "dual_aa":
            optimizer_ul = torch.optim.AdamW(
                model.parameters(), args.unlearn_lr, weight_decay=0.05
            )
            optimizer = torch.optim.AdamW(
                model.parameters(), args.retain_lr, weight_decay=0.05
            )

        if args.imagenet_arch and args.unlearn == "retrain":
            lambda0 = (
                lambda cur_iter: (cur_iter + 1) / args.warmup
                if cur_iter < args.warmup
                else (
                    0.5
                    * (
                        1.0
                        + np.cos(
                            np.pi
                            * (
                                (cur_iter - args.warmup)
                                / (args.unlearn_epochs - args.warmup)
                            )
                        )
                    )
                )
            )
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda0)
            if "dual" in args.optim:
                scheduler_ul = torch.optim.lr_scheduler.LambdaLR(
                    optimizer_ul, lr_lambda=lambda0
                )
            else:
                scheduler_ul = None
        else:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=decreasing_lr, gamma=0.1
            )  # 0.1 is fixed

            if "dual" in args.optim:
                scheduler_ul = torch.optim.lr_scheduler.MultiStepLR(
                    optimizer_ul, milestones=decreasing_lr, gamma=0.1
                )  # 0.1 is fixed
            else:
                scheduler_ul = None
        if args.rewind_epoch != 0:
            # learning rate rewinding
            for _ in range(args.rewind_epoch):
                scheduler.step()
        for epoch in range(0, args.unlearn_epochs):
            start_time = time.time()

            print(
                "Epoch #{}, Learning rate: {}".format(
                    epoch, optimizer.state_dict()["param_groups"][0]["lr"]
                )
            )

            train_acc = unlearn_iter_func(
                data_loaders,
                model,
                criterion,
                optimizer,
                epoch,
                args,
                mask,
                optimizer_ul=optimizer_ul,
                **kwargs,
            )
            if scheduler_ul is not None:
                scheduler_ul.step()
            scheduler.step()

            torch.save(
                model.state_dict(),
                os.path.join(args.save_dir, f"checkpoint_{epoch + 1}.pt"),
            )

            print("one epoch duration:{}".format(time.time() - start_time))

    return _wrapped


def _iterative_scrub_impl(unlearn_iter_func):
    def _wrapped(data_loaders, model, criterion, args, mask=None, **kwargs):
        decreasing_lr = list(map(int, args.decreasing_lr.split(",")))

        # clone model as teacher model
        model_dict = model.state_dict()
        teacher_model = utils.setup_model(args)
        teacher_model.cuda()
        teacher_model.load_state_dict(model_dict)
        for param in teacher_model.parameters():
            param.requires_grad = False

        if args.rewind_epoch != 0:
            initialization = torch.load(
                args.rewind_pth, map_location=torch.device("cuda:" + str(args.gpu))
            )
            current_mask = extract_mask(model.state_dict())
            remove_prune(model)
            # weight rewinding
            # rewind, initialization is a full model architecture without masks
            model.load_state_dict(initialization, strict=True)
            prune_model_custom(model, current_mask)

        optimizer_ul = None
        if args.optim == "sgd":
            optimizer = torch.optim.SGD(
                model.parameters(),
                args.unlearn_lr,
                momentum=args.momentum,
                weight_decay=args.weight_decay,
                # nesterov=True
            )
        elif args.optim == "adam":
            optimizer = torch.optim.AdamW(
                model.parameters(), args.unlearn_lr, weight_decay=0.05
            )
        elif args.optim == "dual" or args.optim == "dual_as":
            optimizer_ul = torch.optim.AdamW(
                model.parameters(), args.unlearn_lr, weight_decay=0.05
            )
            optimizer = torch.optim.SGD(
                model.parameters(),
                args.retain_lr,
                momentum=args.momentum,
                weight_decay=args.weight_decay,
            )
        elif args.optim == "dual_aa":
            optimizer_ul = torch.optim.AdamW(
                model.parameters(), args.unlearn_lr, weight_decay=0.05
            )
            optimizer = torch.optim.AdamW(
                model.parameters(), args.retain_lr, weight_decay=0.05
            )

        if args.imagenet_arch and args.unlearn == "retrain":
            lambda0 = (
                lambda cur_iter: (cur_iter + 1) / args.warmup
                if cur_iter < args.warmup
                else (
                    0.5
                    * (
                        1.0
                        + np.cos(
                            np.pi
                            * (
                                (cur_iter - args.warmup)
                                / (args.unlearn_epochs - args.warmup)
                            )
                        )
                    )
                )
            )
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda0)
        else:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=decreasing_lr, gamma=0.1
            )  # 0.1 is fixed
        if args.rewind_epoch != 0:
            # learning rate rewinding
            for _ in range(args.rewind_epoch):
                scheduler.step()
        for epoch in range(0, args.unlearn_epochs):
            start_time = time.time()

            print(
                "Epoch #{}, Learning rate: {}".format(
                    epoch, optimizer.state_dict()["param_groups"][0]["lr"]
                )
            )

            train_acc = unlearn_iter_func(
                data_loaders,
                model,
                teacher_model,
                criterion,
                optimizer,
                epoch,
                args,
                mask,
                optimizer_ul,
                **kwargs,
            )
            scheduler.step()

            torch.save(
                model.state_dict(),
                os.path.join(args.save_dir, f"checkpoint_{epoch + 1}.pt"),
            )

            print("one epoch duration:{}".format(time.time() - start_time))

    return _wrapped


def iterative_unlearn(func):
    """usage:

    @iterative_unlearn

    def func(data_loaders, model, criterion, optimizer, epoch, args)"""
    return _iterative_unlearn_impl(func)


def iterative_scrub(func):
    """usage:

    @iterative_unlearn

    def func(data_loaders, model, criterion, optimizer, epoch, args)"""
    return _iterative_scrub_impl(func)


def _iterative_unlearn_impl_sam(unlearn_iter_func):
    def _wrapped(data_loaders, model, criterion, args, mask=None, **kwargs):
        decreasing_lr = list(map(int, args.decreasing_lr.split(",")))
        if args.rewind_epoch != 0:
            initialization = torch.load(
                args.rewind_pth, map_location=torch.device("cuda:" + str(args.gpu))
            )
            current_mask = extract_mask(model.state_dict())
            remove_prune(model)
            # weight rewinding
            # rewind, initialization is a full model architecture without masks
            model.load_state_dict(initialization, strict=True)
            prune_model_custom(model, current_mask)

        # optimizer = torch.optim.SGD(
        #     model.parameters(),
        #     args.unlearn_lr,
        #     momentum=args.momentum,
        #     weight_decay=args.weight_decay,
        # )
        optimizer = SAM(
            model.parameters(),
            torch.optim.SGD,
            lr=args.unlearn_lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )

        if args.imagenet_arch and args.unlearn == "retrain":
            lambda0 = (
                lambda cur_iter: (cur_iter + 1) / args.warmup
                if cur_iter < args.warmup
                else (
                    0.5
                    * (
                        1.0
                        + np.cos(
                            np.pi
                            * (
                                (cur_iter - args.warmup)
                                / (args.unlearn_epochs - args.warmup)
                            )
                        )
                    )
                )
            )
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda0)
        else:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=decreasing_lr, gamma=0.1
            )  # 0.1 is fixed
        if args.rewind_epoch != 0:
            # learning rate rewinding
            for _ in range(args.rewind_epoch):
                scheduler.step()
        for epoch in range(0, args.unlearn_epochs):
            start_time = time.time()

            print(
                "Epoch #{}, Learning rate: {}".format(
                    epoch, optimizer.state_dict()["param_groups"][0]["lr"]
                )
            )

            train_acc = unlearn_iter_func(
                data_loaders, model, criterion, optimizer, epoch, args, mask, **kwargs
            )
            scheduler.step()

            print("one epoch duration:{}".format(time.time() - start_time))

    return _wrapped


def iterative_unlearn_sam(func):
    """usage:

    @iterative_unlearn

    def func(data_loaders, model, criterion, optimizer, epoch, args)"""
    return _iterative_unlearn_impl_sam(func)
