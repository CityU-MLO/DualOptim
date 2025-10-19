import math
import torch


def noise_estimation_loss(
    model,
    x0: torch.Tensor,
    t: torch.LongTensor,
    e: torch.Tensor,
    b: torch.Tensor,
    keepdim=False,
):
    a = (1 - b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    x = x0 * a.sqrt() + e * (1.0 - a).sqrt()
    output = model(x, t.float())
    if keepdim:
        return (e - output).square().sum(dim=(1, 2, 3))
    else:
        return (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0)


def noise_estimation_loss_conditional(
    model,
    x0: torch.Tensor,
    t: torch.LongTensor,
    c: torch.LongTensor,
    e: torch.Tensor,
    b: torch.Tensor,
    cond_drop_prob=0.1,
    keepdim=False,
):
    a = (1 - b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    x = x0 * a.sqrt() + e * (1.0 - a).sqrt()
    output = model(x, t.float(), c, cond_drop_prob=cond_drop_prob, mode="train")
    if keepdim:
        return (e - output).square().sum(dim=(1, 2, 3))
    else:
        return (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0)


loss_registry = {"simple": noise_estimation_loss}

loss_registry_conditional = {"simple": noise_estimation_loss_conditional}


def adaptive_loss(
    loss_function,
    model,
    x0: torch.Tensor,
    t: torch.LongTensor,
    c: torch.LongTensor,
    e: torch.Tensor,
    b: torch.Tensor,
    cond_drop_prob=0.1,
    lambd=1,
    keepdim=False,
):
    ori_loss = loss_function(model, x0, t, c, e, b, cond_drop_prob, keepdim=True)
    size = x0.shape[0]
    coef = 1 / (torch.pow(ori_loss.detach().clone(), lambd) + 1e-8)
    ad_loss = (coef / coef.sum()) * ori_loss * size

    if keepdim:
        return ad_loss
    else:
        return ad_loss.mean(dim=0)


def cosine_lr_scheduler(base_lr, current_epoch, T_max):
    return base_lr * (1 + math.cos(math.pi * current_epoch / T_max)) / 2
