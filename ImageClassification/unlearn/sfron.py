from copy import deepcopy
import os
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

import utils

# from .unlearn_method import UnlearnMethod
from trainer import validate
from optim import HybridAdamSGD, DecoupledAdam, DecoupledAdamW
from optim8bit import AdamWDecouple8bit
from lion_pytorch import Lion


class UnlearnMethod:
    def __init__(self, model, loss_function, save_path, args) -> None:
        self.unlearn_dataloaders = None
        self.model = model
        self.loss_function = loss_function
        self.save_path = save_path
        self.args = args
        self.params = {}

    def prepare_unlearn(self, unlearn_dataloaders: dict) -> None:
        self.unlearn_dataloaders = unlearn_dataloaders
        return

    def get_unlearned_model(self) -> nn.Module:
        return self.model

    def get_params(self) -> dict:
        return self.params


def cycle(dl):
    while True:
        for data in dl:
            yield data


def calc_sparsity(tensor):
    # Count zero elements
    num_zero_elements = tensor.numel() - torch.count_nonzero(tensor)

    # Total number of elements
    total_elements = tensor.numel()

    # Compute sparsity
    sparsity = num_zero_elements / total_elements
    return sparsity.item(), total_elements, num_zero_elements


@torch.no_grad()
def update_parameters(source_model, input_model, avg_fn):
    source_param = source_model.parameters()
    input_param = input_model.parameters()
    for p_swa, p_model in zip(source_param, input_param):
        device = p_swa.device
        p_model_ = p_model.detach().to(device)
        p_swa.detach().copy_(avg_fn(p_swa.detach(), p_model_))


def expdecay_lr_scheduler(base_lr, current_epoch, T_max, base=2):
    return base_lr * (1 - current_epoch / T_max) ** base


def linear_lr_scheduler(base_lr, current_epoch, T_max, base=1):
    return base_lr * (1 - current_epoch / T_max) ** base


def cosine_lr_scheduler(base_lr, current_epoch, T_max):
    return base_lr * (1 + math.cos(math.pi * current_epoch / T_max)) / 2


def cosine_warmup_lr_scheduler(base_lr, start_lr, current_epoch, T_max, T_warm):
    if current_epoch < T_warm:
        return start_lr + (base_lr - start_lr) * current_epoch / T_warm
    return base_lr * (1 + math.cos(math.pi * current_epoch / (T_max - T_warm))) / 2


class AdaptiveLoss(torch.nn.Module):
    def __init__(self, loss_function, lambd=1.0, reduction="mean"):
        super(AdaptiveLoss, self).__init__()
        self.loss_function = loss_function  # reduction=none
        self.lambd = lambd
        self.reduction = reduction

    def forward(self, predict, target):
        ori_loss = self.loss_function(predict, target)
        coef = 1 / (torch.pow(ori_loss.detach().clone(), self.lambd) + 1e-15)
        # print((coef / coef.sum()).sum(), ((coef / coef.sum())* predict.shape[0]).sum())
        ad_loss = (coef / coef.sum()) * ori_loss * predict.shape[0]

        if self.reduction == "mean":
            ad_loss = ad_loss.mean()
        elif self.reduction == "sum":
            ad_loss = ad_loss.sum()
        return ad_loss


class SFRon(UnlearnMethod):
    def __init__(self, model, loss_function, save_path, args) -> None:
        super().__init__(model, loss_function, save_path, args)
        self.num_classes = args.num_classes
        self.seed = args.seed
        self.eval = True
        self.forget_loss_function = None
        self.retain_loss_function = None
        self.weight_saliency_mask = None
        self.avg_fn = None
        # params
        # tinyimage 10%
        # self.opt = 'adam'
        # self.momentum = 0.9
        # self.weight_decay = 5e-2
        # self.retain_lr = 2e-5

        # self.n_iters = 500
        # self.unlearn_loss = "adaga"
        # self.forget_freq = 1

        # self.forget_alpha = 500.0
        # self.max_norm = 7.0

        # self.ema_enabled = True
        # self.ema_beta = 1.0

        # self.sched = 'cosine'
        # self.lambd = 0.6

        # self.mask = True
        # self.th = 1

        # self.log_freq = 100
        # CIFAR10 10%
        self.opt = args.optim  # 'sgd'
        self.momentum = args.momentum
        self.weight_decay = args.weight_decay  # 5e-4
        self.retain_lr = args.retain_lr  # 0.01
        self.forget_lr = args.unlearn_lr

        self.n_iters = args.sfron_iters  # 1500
        self.unlearn_loss = "adaga"

        self.forget_freq = args.sfron_freq  # 5
        self.forget_alpha = args.sfron_alpha  # 25

        self.max_norm = 7.0

        self.ema_enabled = True
        self.ema_beta = 1.0

        self.sched = "cosine"
        self.lambd = (
            0.5 if args.dataset != "TinyImagenet" else 0.6
        )  # lambda in the paper

        self.mask = True
        self.th = 1  # gamma in the paper

        self.log_freq = 500 if args.dataset != "TinyImagenet" else 100

        self.num_classes = args.num_classes

    def prepare_unlearn(self, unlearn_dataloaders: dict) -> None:
        self.unlearn_dataloaders = unlearn_dataloaders

        def avg_fn(averaged_model_parameter, model_parameter):
            return (
                1 - self.ema_beta
            ) * model_parameter + self.ema_beta * averaged_model_parameter

        self.avg_fn = avg_fn
        # adaptive gradient ascent
        if self.unlearn_loss == "adaga":
            # adaptive ce loss
            ce_loss = nn.CrossEntropyLoss(reduction="none")
            self.forget_loss_function = AdaptiveLoss(ce_loss, lambd=self.lambd)
        elif self.unlearn_loss == "ga":
            self.forget_loss_function = self.loss_function
        self.retain_loss_function = self.loss_function
        # forget-remain balanced weight saliency mask
        # forget_trainloader = self.unlearn_dataloaders['forget_train']
        # retain_trainloader = self.unlearn_dataloaders['retain_train']
        forget_trainloader = self.unlearn_dataloaders["forget"]
        retain_trainloader = self.unlearn_dataloaders["retain"]

        if self.mask:
            self.weight_saliency_mask = self.get_weight_saliency_mask(
                forget_loader=forget_trainloader,
                remain_loader=retain_trainloader,
                threshold=self.th,
            )
        else:
            self.weight_saliency_mask = None

    def get_unlearned_model(self):
        # retain_trainloader = self.unlearn_dataloaders['retain_train']
        # forget_trainloader = self.unlearn_dataloaders['forget_train']
        retain_trainloader = self.unlearn_dataloaders["retain"]
        forget_trainloader = self.unlearn_dataloaders["forget"]
        retain_train_iter = cycle(retain_trainloader)
        forget_train_iter = cycle(forget_trainloader)

        # retain_validloader = self.unlearn_dataloaders['retain_valid']
        # forget_validloader = self.unlearn_dataloaders['forget_valid']

        if self.sched == "cosine":
            lr_scheduler = cosine_lr_scheduler
        elif self.sched == "linear":
            lr_scheduler = linear_lr_scheduler
        elif self.sched == "expdecay":
            lr_scheduler = expdecay_lr_scheduler
        elif self.sched == "cosine_warmup":
            lr_scheduler = cosine_warmup_lr_scheduler

        optimizer_f = None
        scheduler_f = None
        if self.opt == "sgd":
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                self.retain_lr,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
            )
        elif self.opt == "adam":
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                self.retain_lr,
                betas=(self.momentum, 0.999),
                weight_decay=0.05,
            )
        elif self.opt == "dual_as" or self.opt == "dual":
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                self.retain_lr,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
            )
            optimizer_f = torch.optim.AdamW(
                self.model.parameters(), self.forget_lr, weight_decay=0.05
            )
            scheduler_f = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer_f, T_max=self.n_iters
            )
        elif self.opt == "dual_aa":
            optimizer = torch.optim.AdamW(
                self.model.parameters(), self.retain_lr, weight_decay=0.05
            )
            optimizer_f = torch.optim.AdamW(
                self.model.parameters(), self.forget_lr, weight_decay=0.05
            )
            scheduler_f = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer_f, T_max=self.n_iters
            )
        elif self.opt == "dual_aa_8bit":
            optimizer = AdamWDecouple8bit(
                self.model.parameters(),
                lr=1e-4,
                weight_decay=0.05,
                lr_ratio_1=self.forget_lr / 1e-4,
                lr_ratio_2=self.retain_lr / 1e-4,
                decouple_m=True,
                decouple_v=True,
                switch_freq=self.forget_freq,
            )
        elif self.opt == "dual_sa":
            optimizer = torch.optim.AdamW(
                self.model.parameters(), self.retain_lr, weight_decay=0.05
            )
            optimizer_f = torch.optim.SGD(
                self.model.parameters(),
                self.forget_lr,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
            )
            scheduler_f = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer_f, T_max=self.n_iters
            )
        elif self.opt == "dual_ss":
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                self.retain_lr,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
            )
            optimizer_f = torch.optim.SGD(
                self.model.parameters(),
                self.forget_lr,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
            )
            scheduler_f = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer_f, T_max=self.n_iters
            )
        elif self.opt == "dual_ls":
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                self.retain_lr,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
            )
            optimizer_f = Lion(
                self.model.parameters(),
                self.forget_lr,
                weight_decay=1e-2,
                use_triton=True,
            )
            scheduler_f = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer_f, T_max=self.n_iters
            )
        elif self.opt == "dual_ll":
            optimizer = Lion(
                self.model.parameters(),
                self.retain_lr,
                weight_decay=1e-2,
                use_triton=True,
            )
            optimizer_f = Lion(
                self.model.parameters(),
                self.forget_lr,
                weight_decay=1e-2,
                use_triton=True,
            )
            scheduler_f = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer_f, T_max=self.n_iters
            )
        elif self.opt == "dual_sl":
            optimizer = Lion(
                self.model.parameters(),
                self.retain_lr,
                weight_decay=1e-2,
                use_triton=True,
            )
            optimizer_f = torch.optim.SGD(
                self.model.parameters(),
                self.forget_lr,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
            )
            scheduler_f = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer_f, T_max=self.n_iters
            )
        elif self.opt == "dual_share":
            optimizer = HybridAdamSGD(
                self.model.parameters(),
                lr_sgd=self.retain_lr,
                lr_adam=self.forget_lr,
                weight_decay=self.weight_decay,
            )
        elif self.opt == "dual_aa_m":
            optimizer = DecoupledAdamW(
                self.model.parameters(),
                lr_forget=self.forget_lr,
                lr_retain=self.retain_lr,
                weight_decay=0.05,
                decouple_m=True,
            )
        elif self.opt == "dual_aa_v":
            optimizer = DecoupledAdamW(
                self.model.parameters(),
                lr_forget=self.forget_lr,
                lr_retain=self.retain_lr,
                weight_decay=0.05,
                decouple_v=True,
            )
        else:
            raise ValueError(f"Invalid optimizer {self.opt}")

        if (
            self.opt != "dual_share"
            and self.opt != "dual_aa_m"
            and self.opt != "dual_aa_v"
        ):
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.n_iters
            )

        if self.ema_enabled:
            ori_model = deepcopy(self.model)

        # Variables for monitoring/logging purposes:
        train_steps = 0
        log_forget_steps = 0
        log_remain_steps = 0
        running_forget_loss = 0
        running_forget_acc1 = 0
        running_remain_loss = 0
        running_remain_acc1 = 0
        start_time = time.time()

        grad_var_forget = []
        grad_var_retain = []
        grad_mean_forget = []
        grad_mean_retain = []
        retain_losses = []
        forget_losses = []
        momentum_norm1 = []
        momentum_norm2 = []
        grad_norm_f = []
        similarity_f = []
        grad_norm_r = []
        similarity_r = []
        sim_fr = []
        momentum_fr_sim = []

        prev_grad = None
        grad_sim = []
        prev_grad1 = None
        grad_sim1 = []
        prev_momentum = None
        momentum_sim = []
        prev_momentum1 = None
        momentum_sim1 = []
        for step in range(0, self.n_iters):
            self.model.train()
            if step % self.forget_freq == 0:
                cur_forget_alpha = lr_scheduler(self.forget_alpha, step, self.n_iters)
                # cur_forget_alpha = self.forget_alpha
                # adaptive weighted gradient ascent on forget train
                x_forget, y_forget = next(forget_train_iter)
                x_forget, y_forget = x_forget.cuda(), y_forget.cuda()
                optimizer.zero_grad() if optimizer_f is None else optimizer_f.zero_grad()
                outputs = self.model(x_forget)
                ori_forget_loss = -self.forget_loss_function(outputs, y_forget)
                forget_loss = cur_forget_alpha * ori_forget_loss
                forget_loss.backward()

                # forget_losses.append(forget_loss.item() / cur_forget_alpha)

                if self.mask:
                    for name, param in self.model.named_parameters():
                        if param.grad is not None:
                            param.grad *= self.weight_saliency_mask[name].to(
                                param.grad.device
                            )
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=self.max_norm
                )
                if self.opt == "dual_share":
                    optimizer.step_adam()
                elif self.opt == "dual_aa_m" or self.opt == "dual_aa_v":
                    optimizer.step_forget()
                else:
                    optimizer.step() if optimizer_f is None else optimizer_f.step()

                grads = []
                with torch.no_grad():
                    for name, param in self.model.named_parameters():
                        if param.grad is not None:
                            grads.append(param.grad.clone().view(-1) / cur_forget_alpha)
                grads_f = torch.cat(grads)
                # if prev_grad is not None:
                #     grad_sim.append(torch.nn.functional.cosine_similarity(prev_grad, grads_f, dim=0).item())
                # prev_grad = grads_f
                # grad_var_forget.append(grads_f.var().item())
                # grad_mean_forget.append(grads_f.mean().item())
                grad_norm_f.append(grads_f.norm().item())

                # momentum = []
                # with torch.no_grad():
                #     state = optimizer.state_dict() if optimizer_f is None else optimizer_f.state_dict()
                #     state = state['state']
                #     for key, value in state.items():
                #         momentum.append(value['momentum_buffer'].clone().view(-1))
                # momentum = torch.cat(momentum)
                # if prev_momentum is not None:
                #     momentum_sim.append(torch.nn.functional.cosine_similarity(prev_momentum, momentum, dim=0).item())
                # prev_momentum = momentum
                # momentum_norm1.append(momentum.norm().item())
                # similarity_f.append(torch.nn.functional.cosine_similarity(momentum, grads_f, dim=0).item())

                forget_acc1 = utils.accuracy(outputs.data, y_forget)[0]
                running_forget_loss += ori_forget_loss
                running_forget_acc1 += forget_acc1
                log_forget_steps += 1

            # train on retain train
            self.model.train()
            x_retain, y_retain = next(retain_train_iter)
            x_retain, y_retain = x_retain.cuda(), y_retain.cuda()

            optimizer.zero_grad()
            outputs = self.model(x_retain)
            ori_remain_loss = self.retain_loss_function(outputs, y_retain)
            remain_loss = ori_remain_loss
            remain_loss.backward()

            # retain_losses.append(remain_loss.item())
            if self.opt == "dual_share":
                optimizer.step_sgd()
            elif self.opt == "dual_aa_m" or self.opt == "dual_aa_v":
                optimizer.step_retain()
            else:
                optimizer.step()

            if step % self.forget_freq == self.forget_freq - 1:
                grads = []
                with torch.no_grad():
                    for name, param in self.model.named_parameters():
                        if param.grad is not None:
                            grads.append(param.grad.clone().view(-1))
                grads_r = torch.cat(grads)
                # sim_fr.append(torch.nn.functional.cosine_similarity(grads_f, grads_r, dim=0).item())
                # if prev_grad1 is not None:
                #     grad_sim1.append(torch.nn.functional.cosine_similarity(prev_grad1, grads_r, dim=0).item())
                # prev_grad1 = grads_r
                grad_norm_r.append(grads_r.norm().item())
                # grad_var_retain.append(grads_r.var().item())
                # grad_mean_retain.append(grads_r.mean().item())

                # momentum = []
                # state = optimizer.state_dict()
                # state = state['state']
                # for key, value in state.items():
                #     momentum.append(value['momentum_buffer'].clone().view(-1))
                # momentum = torch.cat(momentum)
                # if prev_momentum1 is not None:
                #     momentum_sim1.append(
                #         torch.nn.functional.cosine_similarity(prev_momentum1, momentum, dim=0).item())
                # prev_momentum1 = momentum
                # momentum_norm2.append(momentum.norm().item())
                # similarity_r.append(torch.nn.functional.cosine_similarity(momentum, grads_r, dim=0).item())

            remain_acc1 = utils.accuracy(outputs.detach().data, y_retain)[0]
            running_remain_loss += ori_remain_loss.detach()
            running_remain_acc1 += remain_acc1

            log_remain_steps += 1
            if self.opt == "dual_share":
                lr = optimizer.param_groups[0]["lr_sgd"]
            elif self.opt == "dual_aa_m" or self.opt == "dual_aa_v":
                lr = optimizer.param_groups[0]["lr_retain"]
            else:
                lr = optimizer.param_groups[0]["lr"]
            train_steps += 1

            if self.eval and train_steps % self.log_freq == 0:
                # Measure training speed:
                end_time = time.time()
                steps_per_sec = log_remain_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_forget_loss = (
                    running_forget_loss.clone().detach() / log_forget_steps
                ).item()
                avg_forget_acc1 = (
                    running_forget_acc1.clone().detach() / log_forget_steps
                ).item()
                avg_remain_loss = (
                    running_remain_loss.clone().detach() / log_remain_steps
                ).item()
                avg_remain_acc1 = (
                    running_remain_acc1.clone().detach() / log_remain_steps
                ).item()
                print(
                    f"step={train_steps} Forget L:{avg_forget_loss:.4f} Acc:{avg_forget_acc1:.4f} Remain L:{avg_remain_loss:.4f} Acc:{avg_remain_acc1:.4f} LR:{lr}"
                )
                # validata on retain valid, forget train, forget valid
                # validate(forget_trainloader, self.model, self.loss_function, "Forget Train")
                # if retain_validloader:
                #     validate(retain_validloader, self.model, self.loss_function, "Retain Valid")
                # if forget_validloader:
                #     validate(forget_validloader, self.model, self.loss_function, "Forget Valid")
                # Reset monitoring variables:
                log_forget_steps = 0
                log_remain_steps = 0
                running_forget_loss = 0
                running_forget_acc1 = 0
                running_remain_loss = 0
                running_remain_acc1 = 0
                start_time = time.time()

            if self.ema_enabled:
                update_parameters(self.model, ori_model, self.avg_fn)
                ori_model = deepcopy(self.model)

            if (
                self.opt != "dual_share"
                and self.opt != "dual_aa_m"
                and self.opt != "dual_aa_v"
            ):
                scheduler.step()
                if scheduler_f is not None:
                    scheduler_f.step()
            elif self.opt == "dual_aa_m" or self.opt == "dual_aa_v":
                current_lr_f = cosine_lr_scheduler(self.forget_lr, step, self.n_iters)
                current_lr_r = cosine_lr_scheduler(self.retain_lr, step, self.n_iters)
                for param_group in optimizer.param_groups:
                    param_group["lr_forget"] = current_lr_f
                    param_group["lr_retain"] = current_lr_r
            else:
                current_lr_f = cosine_lr_scheduler(self.forget_lr, step, self.n_iters)
                current_lr_r = cosine_lr_scheduler(self.retain_lr, step, self.n_iters)
                for param_group in optimizer.param_groups:
                    param_group["lr_adam"] = current_lr_f
                    param_group["lr_sgd"] = current_lr_r

            # if (step+1) % (6*self.forget_freq) == 0:
            #     # save checkpoint
            #     torch.save(self.model.state_dict(), os.path.join(self.save_path, f"checkpoint_{step+1}.pt"))
        return (
            self.model,
            grad_var_forget,
            grad_var_retain,
            grad_mean_forget,
            grad_mean_retain,
            retain_losses,
            forget_losses,
            momentum_norm1,
            momentum_norm2,
            grad_norm_f,
            grad_norm_r,
            similarity_f,
            similarity_r,
            grad_sim,
            grad_sim1,
            momentum_sim,
            momentum_sim1,
            sim_fr,
            momentum_fr_sim,
        )
        # return self.model

    def get_weight_saliency_mask(self, forget_loader, remain_loader, threshold):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0)
        criterion = self.loss_function

        # forget fisher
        forget_fisher_path = os.path.join(self.save_path, "forget_fisher.pt")
        if os.path.exists(forget_fisher_path):
            forget_gradients = torch.load(forget_fisher_path, weights_only=True)
        else:
            forget_gradients = {}
            for name, param in self.model.named_parameters():
                forget_gradients[name] = 0
            self.model.eval()
            for i, (image, target) in enumerate(forget_loader):
                image = image.cuda()
                target = target.cuda()

                # compute output
                output_clean = self.model(image)
                loss = criterion(output_clean, target)

                optimizer.zero_grad()
                loss.backward()

                with torch.no_grad():
                    for name, param in self.model.named_parameters():
                        if param.grad is not None:
                            forget_gradients[name] += param.grad.data.cpu() ** 2 / len(
                                forget_loader
                            )

            torch.save(forget_gradients, forget_fisher_path)

        # remain fisher
        remain_fisher_path = os.path.join(self.save_path, "remain_fisher.pt")
        if os.path.exists(remain_fisher_path):
            remain_gradients = torch.load(remain_fisher_path, weights_only=True)
        else:
            remain_gradients = {}
            for name, param in self.model.named_parameters():
                remain_gradients[name] = 0
            self.model.eval()
            for i, (image, target) in enumerate(remain_loader):
                image = image.cuda()
                target = target.cuda()

                # compute output
                output_clean = self.model(image)
                loss = criterion(output_clean, target)

                optimizer.zero_grad()
                loss.backward()

                with torch.no_grad():
                    for name, param in self.model.named_parameters():
                        if param.grad is not None:
                            remain_gradients[name] += param.grad.data.cpu() ** 2 / len(
                                remain_loader
                            )

            torch.save(remain_gradients, remain_fisher_path)

        total_cnt = 0
        w_cnt = 0
        weight_saliency_mask = {}
        for name in forget_gradients.keys():
            weight_saliency_mask[name] = 0
            try:
                weight_saliency = (forget_gradients[name] + 1e-15) / (
                    remain_gradients[name] + 1e-15
                )
                w = weight_saliency >= threshold
                w_sparsity, total_elements, w_num_zero_elements = calc_sparsity(w)
                total_cnt += total_elements
                w_cnt += w_num_zero_elements
                weight_saliency_mask[name] = w
            except:
                pass
        print(f"Total sparsity th:{threshold} weight:{w_cnt/total_cnt*100}")
        return weight_saliency_mask

    def get_params(self) -> dict:
        self.params = {
            "opt": self.opt,
            "momentum": self.momentum,
            "weight_decay": self.weight_decay,
            "retain_lr": self.retain_lr,
            "n_iters": self.n_iters,
            "forget_freq": self.forget_freq,
            "forget_alpha": self.forget_alpha,
            "max_norm": self.max_norm,
            "ema_beta": self.ema_beta,
            "sched": self.sched,
            "lambd": self.lambd,
            "mask": self.mask,
            "threshold": self.th,
        }
        return self.params
