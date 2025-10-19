import torch


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (
                    (torch.pow(p, 2) if group["adaptive"] else 1.0)
                    * p.grad
                    * scale.to(p)
                )
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert (
            closure is not None
        ), "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(
            closure
        )  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][
            0
        ].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
            torch.stack(
                [
                    ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad)
                    .norm(p=2)
                    .to(shared_device)
                    for group in self.param_groups
                    for p in group["params"]
                    if p.grad is not None
                ]
            ),
            p=2,
        )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups


class Lion(torch.optim.Optimizer):
    r"""Implements Lion algorithm."""

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        """Initialize the hyperparameters.
    Args:
      params (iterable): iterable of parameters to optimize or dicts defining
        parameter groups
      lr (float, optional): learning rate (default: 1e-4)
      betas (Tuple[float, float], optional): coefficients used for computing
        running averages of gradient and its square (default: (0.9, 0.99))
      weight_decay (float, optional): weight decay coefficient (default: 0)
    """

        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
    Args:
      closure (callable, optional): A closure that reevaluates the model
        and returns the loss.
    Returns:
      the loss.
    """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                # Perform stepweight decay
                p.data.mul_(1 - group["lr"] * group["weight_decay"])

                grad = p.grad
                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p)

                exp_avg = state["exp_avg"]
                beta1, beta2 = group["betas"]

                # Weight update
                update = exp_avg * beta1 + grad * (1 - beta1)
                p.add_(torch.sign(update), alpha=-group["lr"])
                # Decay the momentum running average coefficient
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)

        return loss


class HybridAdamSGD(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr_sgd=0.1,
        lr_adam=0.001,
        beta1=0.9,
        beta2=0.999,
        weight_decay=0,
        eps=1e-8,
    ):
        defaults = dict(
            lr_sgd=lr_sgd,
            lr_adam=lr_adam,
            beta1=beta1,
            beta2=beta2,
            weight_decay=weight_decay,
            eps=eps,
        )
        super(HybridAdamSGD, self).__init__(params, defaults)

        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["m"] = torch.zeros_like(p.data)
                state["v"] = torch.zeros_like(p.data)
                state["t"] = 0

    @torch.no_grad()
    def step_sgd(self):
        for group in self.param_groups:
            lr_sgd = group["lr_sgd"]
            momentum = group["beta1"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if weight_decay != 0:
                    grad.add_(p.data, alpha=weight_decay)

                state = self.state[p]
                velocity = state["m"]
                # velocity.mul_(momentum).add_(grad)
                velocity.mul_(momentum).add_(grad, alpha=1 - momentum)
                p.data.add_(velocity, alpha=-lr_sgd)

    @torch.no_grad()
    def step_adam(self):
        for group in self.param_groups:
            lr_adam = group["lr_adam"]
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if weight_decay != 0:
                    grad.add_(p.data, alpha=weight_decay)

                state = self.state[p]
                m, v, t = state["m"], state["v"], state["t"]
                t += 1
                state["t"] = t

                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** t
                bias_correction2 = 1 - beta2 ** t

                step_size = lr_adam / bias_correction1
                denom = (v.sqrt() / (bias_correction2 ** 0.5)).add_(eps)

                p.data.addcdiv_(m, denom, value=-step_size)


class DecoupledAdam(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr_forget=0.001,
        lr_retain=0.001,
        beta1=0.9,
        beta2=0.999,
        weight_decay=0,
        eps=1e-8,
        decouple_m=False,
        decouple_v=False,
    ):
        defaults = dict(
            lr_forget=lr_forget,
            lr_retain=lr_retain,
            beta1=beta1,
            beta2=beta2,
            weight_decay=weight_decay,
            eps=eps,
            decouple_m=decouple_m,
            decouple_v=decouple_v,
        )
        super(DecoupledAdam, self).__init__(params, defaults)

        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if group["decouple_m"]:
                    state["m1"] = torch.zeros_like(p.data)
                    state["m2"] = torch.zeros_like(p.data)
                else:
                    state["m"] = torch.zeros_like(p.data)
                if group["decouple_v"]:
                    state["v1"] = torch.zeros_like(p.data)
                    state["v2"] = torch.zeros_like(p.data)
                else:
                    state["v"] = torch.zeros_like(p.data)
                state["t"] = 0
                state["t1"] = 0
                state["t2"] = 0

    @torch.no_grad()
    def step_forget(self):
        for group in self.param_groups:
            lr_adam = group["lr_forget"]
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if weight_decay != 0:
                    grad.add_(p.data, alpha=weight_decay)

                state = self.state[p]

                m = state["m1"] if group["decouple_m"] else state["m"]
                v = state["v1"] if group["decouple_v"] else state["v"]

                t = state["t"]
                t += 1
                state["t"] = t
                t1 = state["t1"]
                t1 += 1
                state["t1"] = t1

                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = (
                    1 - beta1 ** t1 if group["decouple_m"] else 1 - beta1 ** t
                )
                bias_correction2 = (
                    1 - beta2 ** t1 if group["decouple_v"] else 1 - beta2 ** t
                )

                step_size = lr_adam / bias_correction1
                denom = (v.sqrt() / (bias_correction2 ** 0.5)).add_(eps)

                p.data.addcdiv_(m, denom, value=-step_size)

    @torch.no_grad()
    def step_retain(self):
        for group in self.param_groups:
            lr_adam = group["lr_adam"]
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if weight_decay != 0:
                    grad.add_(p.data, alpha=weight_decay)

                state = self.state[p]
                m = state["m1"] if group["decouple_m"] else state["m"]
                v = state["v1"] if group["decouple_v"] else state["v"]

                t = state["t"]
                t += 1
                state["t"] = t
                t2 = state["t2"]
                t2 += 1
                state["t2"] = t2

                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = (
                    1 - beta1 ** t2 if group["decouple_m"] else 1 - beta1 ** t
                )
                bias_correction2 = (
                    1 - beta2 ** t2 if group["decouple_v"] else 1 - beta2 ** t
                )

                step_size = lr_adam / bias_correction1
                denom = (v.sqrt() / (bias_correction2 ** 0.5)).add_(eps)

                p.data.addcdiv_(m, denom, value=-step_size)


class DecoupledAdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr_forget=0.001,
        lr_retain=0.001,
        beta1=0.9,
        beta2=0.999,
        weight_decay=0,
        eps=1e-8,
        decouple_m=False,
        decouple_v=False,
    ):
        defaults = dict(
            lr_forget=lr_forget,
            lr_retain=lr_retain,
            beta1=beta1,
            beta2=beta2,
            weight_decay=weight_decay,
            eps=eps,
            decouple_m=decouple_m,
            decouple_v=decouple_v,
        )
        super(DecoupledAdamW, self).__init__(params, defaults)

        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if group["decouple_m"]:
                    state["m1"] = torch.zeros_like(p.data)
                    state["m2"] = torch.zeros_like(p.data)
                else:
                    state["m"] = torch.zeros_like(p.data)
                if group["decouple_v"]:
                    state["v1"] = torch.zeros_like(p.data)
                    state["v2"] = torch.zeros_like(p.data)
                else:
                    state["v"] = torch.zeros_like(p.data)
                state["t"] = 0
                state["t1"] = 0
                state["t2"] = 0

    # @torch.no_grad()
    def step_forget(self):
        for group in self.param_groups:
            lr = group["lr_forget"]
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                m = state["m1"] if group["decouple_m"] else state["m"]
                v = state["v1"] if group["decouple_v"] else state["v"]

                t = state["t"]
                t += 1
                state["t"] = t
                t1 = state["t1"]
                t1 += 1
                state["t1"] = t1

                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = (
                    1 - beta1 ** t1 if group["decouple_m"] else 1 - beta1 ** t
                )
                bias_correction2 = (
                    1 - beta2 ** t1 if group["decouple_v"] else 1 - beta2 ** t
                )

                step_size = lr / bias_correction1
                denom = (v.sqrt() / (bias_correction2 ** 0.5)).add_(eps)

                if weight_decay != 0:
                    p.data.add_(p.data, alpha=-lr * weight_decay)
                p.data.addcdiv_(m, denom, value=-step_size)

    # @torch.no_grad()
    def step_retain(self):
        for group in self.param_groups:
            lr = group["lr_retain"]
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]
                m = state["m1"] if group["decouple_m"] else state["m"]
                v = state["v1"] if group["decouple_v"] else state["v"]

                t = state["t"]
                t += 1
                state["t"] = t
                t2 = state["t2"]
                t2 += 1
                state["t2"] = t2

                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = (
                    1 - beta1 ** t2 if group["decouple_m"] else 1 - beta1 ** t
                )
                bias_correction2 = (
                    1 - beta2 ** t2 if group["decouple_v"] else 1 - beta2 ** t
                )

                step_size = lr / bias_correction1
                denom = (v.sqrt() / (bias_correction2 ** 0.5)).add_(eps)

                if weight_decay != 0:
                    p.data.add_(p.data, alpha=-lr * weight_decay)
                p.data.addcdiv_(m, denom, value=-step_size)


class AdamDecouple(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr=0.001,
        beta1=0.9,
        beta2=0.999,
        weight_decay=0,
        eps=1e-8,
        decouple_m=True,
        decouple_v=True,
        lr_forget_ratio=1.0,
        lr_retain_ratio=1.0,
    ):
        """
        lr is uniform so that scheduler can be applied
        we adjust the learning rate for forget and retain by the ratio
        """
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= beta1 < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {beta1}")
        if not 0.0 <= beta2 < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {beta2}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0.0 <= lr_forget_ratio:
            raise ValueError(f"Invalid lr_forget_ratio value: {lr_forget_ratio}")
        if not 0.0 <= lr_retain_ratio:
            raise ValueError(f"Invalid lr_retain_ratio value: {lr_retain_ratio}")
        defaults = dict(
            lr=lr,
            beta1=beta1,
            beta2=beta2,
            weight_decay=weight_decay,
            eps=eps,
            decouple_m=decouple_m,
            decouple_v=decouple_v,
            lr_forget_ratio=lr_forget_ratio,
            lr_retain_ratio=lr_retain_ratio,
        )
        super(AdamDecouple, self).__init__(params, defaults)

        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if group["decouple_m"]:
                    state["m1"] = torch.zeros_like(p.data)
                    state["m2"] = torch.zeros_like(p.data)
                else:
                    state["m"] = torch.zeros_like(p.data)
                if group["decouple_v"]:
                    state["v1"] = torch.zeros_like(p.data)
                    state["v2"] = torch.zeros_like(p.data)
                else:
                    state["v"] = torch.zeros_like(p.data)
                state["t"] = 0
                state["t1"] = 0
                state["t2"] = 0

    @torch.no_grad()
    def step(self, closure=None, mode="retain"):
        """
        switch mode to forget or retain
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            ratio = (
                group["lr_forget_ratio"]
                if mode == "forget"
                else group["lr_retain_ratio"]
            )

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if weight_decay != 0:
                    grad.add_(p.data, alpha=weight_decay)
                state = self.state[p]

                if group["decouple_m"]:
                    m = state["m1"] if mode == "forget" else state["m2"]
                else:
                    m = state["m"]

                if group["decouple_v"]:
                    v = state["v1"] if mode == "forget" else state["v2"]
                else:
                    v = state["v"]

                t = state["t"]
                t += 1
                state["t"] = t
                t_prim = state["t1"] if mode == "forget" else state["t2"]
                t_prim += 1
                state["t1" if mode == "forget" else "t2"] = t_prim

                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = (
                    1 - beta1 ** t_prim if group["decouple_m"] else 1 - beta1 ** t
                )
                bias_correction2 = (
                    1 - beta2 ** t_prim if group["decouple_v"] else 1 - beta2 ** t
                )

                step_size = lr / bias_correction1
                denom = (v.sqrt() / (bias_correction2 ** 0.5)).add_(eps)

                p.data.addcdiv_(m, denom, value=-ratio * step_size)
        return loss


class AdamWDecouple(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr=0.001,
        beta1=0.9,
        beta2=0.999,
        weight_decay=0,
        eps=1e-8,
        decouple_m=True,
        decouple_v=True,
        lr_forget_ratio=1.0,
        lr_retain_ratio=1.0,
    ):
        """
        lr is uniform so that scheduler can be applied
        we adjust the learning rate for forget and retain by the ratio
        """
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= beta1 < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {beta1}")
        if not 0.0 <= beta2 < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {beta2}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0.0 <= lr_forget_ratio:
            raise ValueError(f"Invalid lr_forget_ratio value: {lr_forget_ratio}")
        if not 0.0 <= lr_retain_ratio:
            raise ValueError(f"Invalid lr_retain_ratio value: {lr_retain_ratio}")
        defaults = dict(
            lr=lr,
            beta1=beta1,
            beta2=beta2,
            weight_decay=weight_decay,
            eps=eps,
            decouple_m=decouple_m,
            decouple_v=decouple_v,
            lr_forget_ratio=lr_forget_ratio,
            lr_retain_ratio=lr_retain_ratio,
        )
        super(AdamWDecouple, self).__init__(params, defaults)

        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if group["decouple_m"]:
                    state["m1"] = torch.zeros_like(p.data)
                    state["m2"] = torch.zeros_like(p.data)
                else:
                    state["m"] = torch.zeros_like(p.data)
                if group["decouple_v"]:
                    state["v1"] = torch.zeros_like(p.data)
                    state["v2"] = torch.zeros_like(p.data)
                else:
                    state["v"] = torch.zeros_like(p.data)
                state["t"] = 0
                state["t1"] = 0
                state["t2"] = 0

    @torch.no_grad()
    def step(self, closure=None, mode="retain"):
        """
        switch mode to forget or retain
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            ratio = (
                group["lr_forget_ratio"]
                if mode == "forget"
                else group["lr_retain_ratio"]
            )

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                if group["decouple_m"]:
                    m = state["m1"] if mode == "forget" else state["m2"]
                else:
                    m = state["m"]

                if group["decouple_v"]:
                    v = state["v1"] if mode == "forget" else state["v2"]
                else:
                    v = state["v"]

                t = state["t"]
                t += 1
                state["t"] = t
                t_prim = state["t1"] if mode == "forget" else state["t2"]
                t_prim += 1
                state["t1" if mode == "forget" else "t2"] = t_prim

                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = (
                    1 - beta1 ** t_prim if group["decouple_m"] else 1 - beta1 ** t
                )
                bias_correction2 = (
                    1 - beta2 ** t_prim if group["decouple_v"] else 1 - beta2 ** t
                )

                step_size = lr / bias_correction1
                denom = (v.sqrt() / (bias_correction2 ** 0.5)).add_(eps)

                if weight_decay != 0:
                    p.data.mul_(1 - ratio * lr * weight_decay)

                p.data.addcdiv_(m, denom, value=-ratio * step_size)
        return loss
