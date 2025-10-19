import argparse
import os
from datetime import datetime

import torch.optim as optim
import yaml


def get_optimizer(config, parameters):
    if config.optim.optimizer == "Adam":
        return optim.Adam(
            parameters,
            lr=config.optim.lr,
            weight_decay=config.optim.weight_decay,
            betas=(config.optim.beta1, 0.999),
            amsgrad=config.optim.amsgrad,
            eps=config.optim.eps,
        )
    elif config.optim.optimizer == "RMSProp":
        return optim.RMSprop(
            parameters, lr=config.optim.lr, weight_decay=config.optim.weight_decay
        )
    elif config.optim.optimizer == "SGD":
        return optim.SGD(parameters, lr=config.optim.lr, momentum=0.9)
    else:
        raise NotImplementedError(
            "Optimizer {} not understood.".format(config.optim.optimizer)
        )


def get_config_and_setup_dirs(args, filename: str):
    with open(filename, "r") as fp:
        config = yaml.safe_load(fp)
    config = dict2namespace(config)

    timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    config.exp_root_dir = os.path.join(
        os.path.join(args.exp_root, f"results"),
        config.data.dataset.lower(),
        args.mode,
        timestamp,
    )
    config.log_dir = os.path.join(config.exp_root_dir, "logs")
    config.ckpt_dir = os.path.join(config.exp_root_dir, "ckpts")
    os.makedirs(config.log_dir)
    os.makedirs(config.ckpt_dir)

    with open(os.path.join(config.log_dir, "config.yaml"), "w") as fp:
        yaml.dump(config, fp)

    return config


def get_forget_config_and_setup_dirs(args, filename: str):
    with open(filename, "r") as fp:
        config = yaml.safe_load(fp)
    config = dict2namespace(config)

    timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")

    unlearn_loss = (
        f"{args.unlearn_loss}{config.training.gamma}"
        if args.unlearn_loss == "adaga"
        else args.unlearn_loss
    )

    if args.mode == "sfron":
        config.exp_root_dir = os.path.join(
            os.path.join(args.exp_root, f"results"),
            config.data.dataset.lower(),
            f"forget_{args.label_to_forget}",
            f"{args.method}_{unlearn_loss}",
            f"f{args.forget_alpha}{args.decay_forget_alpha}_r{args.remain_alpha}_lr{config.optim.lr}"
            + ("_dual" if args.dual else ""),
            timestamp,
        )
    else:
        config.exp_root_dir = os.path.join(
            os.path.join(args.exp_root, f"results"),
            config.data.dataset.lower(),
            f"forget_{args.label_to_forget}",
            f"{args.mode}",
            f"f{args.forget_alpha}_r{args.remain_alpha}_lr{config.optim.lr}",
            timestamp,
        )
    config.log_dir = os.path.join(config.exp_root_dir, "logs")
    config.ckpt_dir = os.path.join(config.exp_root_dir, "ckpts")
    os.makedirs(config.log_dir)
    os.makedirs(config.ckpt_dir)

    with open(os.path.join(config.log_dir, "config.yaml"), "w") as fp:
        yaml.dump(config, fp)

    return config


def setup_dirs(config, args):
    timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    config.exp_root_dir = os.path.join(
        os.path.join(args.exp_root, f"results"), config.data.dataset.lower(), timestamp
    )
    config.log_dir = os.path.join(config.exp_root_dir, "logs")
    config.ckpt_dir = os.path.join(config.exp_root_dir, "ckpts")
    os.makedirs(config.log_dir)
    os.makedirs(config.ckpt_dir)

    # wandb_id = wandb.util.generate_id()
    # config.wandb_id = wandb_id

    with open(os.path.join(config.exp_root_dir, "config.yaml"), "w") as fp:
        yaml.dump(config, fp)

    return config


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def cycle(dl):
    while True:
        for data in dl:
            yield data


def create_class_labels(string, n_classes=10):
    if any(x.startswith("x") for x in string.split(",")):
        excluded_numbers = [int(x[1:]) for x in string.split(",") if x.startswith("x")]
        my_list = [x for x in range(n_classes) if x not in excluded_numbers]
    else:
        excluded_numbers = []
        my_list = [int(x) for x in string.split(",")]
    return my_list, excluded_numbers
