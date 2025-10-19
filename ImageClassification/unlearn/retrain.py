from trainer import train, adv_train, bd_train

from .impl import iterative_unlearn, iterative_unlearn_sam


@iterative_unlearn
def retrain(
    data_loaders, model, criterion, optimizer, epoch, args, mask, optimizer_ul=None
):
    retain_loader = data_loaders["retain"]
    return train(retain_loader, model, criterion, optimizer, epoch, args, mask)


@iterative_unlearn_sam
def retrain_sam(
    data_loaders, model, criterion, optimizer, epoch, args, mask, optimizer_ul=None
):
    retain_loader = data_loaders["retain"]
    return train(retain_loader, model, criterion, optimizer, epoch, args, mask)


@iterative_unlearn
def retrain_adv(
    data_loaders, model, criterion, optimizer, epoch, args, mask, optimizer_ul=None
):
    retain_loader = data_loaders["retain"]
    return adv_train(
        retain_loader, model, criterion, optimizer, epoch, args, mask, eps=args.eps
    )


@iterative_unlearn
def retrain_bd(
    data_loaders, model, criterion, optimizer, epoch, args, mask, optimizer_ul=None
):
    retain_loader = data_loaders["retain"]
    return bd_train(retain_loader, model, criterion, optimizer, epoch, args, mask)
