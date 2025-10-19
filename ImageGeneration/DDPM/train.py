import argparse
import logging
import os
import sys
import traceback

import numpy as np
import torch
from functions import get_config_and_setup_dirs, get_forget_config_and_setup_dirs
from runners.diffusion import Diffusion

torch.set_printoptions(sci_mode=False)


def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()["__doc__"])
    parser.add_argument(
        "--exp_root", type=str, required=True, help="Path to the experiment root"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the config file"
    )
    parser.add_argument(
        "--ckpt_folder",
        type=str,
        help="Path to folder with pretrained model. Only for forgetting training.",
    )
    parser.add_argument("--mode", type=str, default="train", help="['train', 'forget']")
    parser.add_argument(
        "--label_to_forget",
        type=int,
        default=0,
        help="Class label 0-9 to forget. Only for forgetting training.",
    )
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument(
        "--verbose",
        type=str,
        default="info",
        help="Verbose level: info | debug | warning | critical",
    )
    # parser.add_argument("--use_pretrained", action="store_true")
    parser.add_argument(
        "--sample_type",
        type=str,
        default="generalized",
        help="sampling approach (generalized or ddpm_noisy)",
    )
    parser.add_argument(
        "--skip_type",
        type=str,
        default="uniform",
        help="skip according to (uniform or quadratic)",
    )
    parser.add_argument(
        "--timesteps", type=int, default=1000, help="number of steps involved"
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=1.0,
        help="eta used to control the variances of sigma",
    )
    parser.add_argument(
        "--cond_scale",
        type=float,
        default=2.0,
        help="classifier-free guidance conditional strength",
    )
    parser.add_argument("--sequence", action="store_true")
    parser.add_argument(
        "--uc", type=bool, default=True, help="whether use unconditional loss or not"
    )
    parser.add_argument(
        "--negative_guidance",
        type=float,
        default=7.5,
        help="classifier-free guidance conditional strength",
    )
    # forget
    parser.add_argument(
        "--forget_alpha", type=float, default=1.0, help="forget loss alpha"
    )
    parser.add_argument(
        "--decay_forget_alpha",
        action="store_true",
        help="whether decay forget loss alpha",
    )
    parser.add_argument(
        "--remain_alpha", type=float, default=1.0, help="remain loss alpha"
    )
    parser.add_argument(
        "--unlearn_loss", type=str, default=None, help="the loss to unlearn"
    )
    parser.add_argument(
        "--method", type=str, default=None, help="the method to unlearn"
    )
    parser.add_argument(
        "--mask_path", type=str, default=None, help="the path to store mask"
    )
    parser.add_argument(
        "--mask_ratio", type=float, default=0.5, help="mask ratio for unlearning"
    )
    parser.add_argument(
        "--sparse", type=bool, default=False, help="whether use sparse unlearn or not"
    )
    parser.add_argument(
        "--dual", action="store_true", help="whether use dual optimizers or not"
    )
    parser.add_argument(
        "--forget_lr", type=float, default=1e-4, help="learning rate for forgetting"
    )

    args = parser.parse_args()

    if args.mode == "sfron" or args.mode == "sa" or args.mode == "salun":
        config = get_forget_config_and_setup_dirs(
            args, os.path.join("configs", args.config)
        )
    else:
        config = get_config_and_setup_dirs(args, os.path.join("configs", args.config))

    # setup logger
    level = getattr(logging, args.verbose.upper(), None)
    if not isinstance(level, int):
        raise ValueError("level {} not supported".format(args.verbose))

    handler1 = logging.StreamHandler()
    handler2 = logging.FileHandler(os.path.join(config.log_dir, "stdout.txt"))
    formatter = logging.Formatter(
        "%(levelname)s - %(filename)s - %(asctime)s - %(message)s"
    )
    handler1.setFormatter(formatter)
    handler2.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    logger.setLevel(level)

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = True

    return args, config


def main():
    args, config = parse_args_and_config()
    if args.mode == "pretrain":
        logging.info("Writing log file to {}".format(config.log_dir))
    try:
        if args.mode == "pretrain":
            runner = Diffusion(args, config)
            runner.train()
        elif args.mode == "retrain":
            runner = Diffusion(args, config)
            runner.retrain()
        elif args.mode == "sfron":
            runner = Diffusion(args, config)
            runner.sfron_forget()
        elif args.mode == "sa":
            runner = Diffusion(args, config)
            runner.sa_forget()
        elif args.mode == "salun":
            runner = Diffusion(args, config)
            runner.saliency_unlearn()
        elif args.mode == "generate_mask":
            runner = Diffusion(args, config)
            runner.generate_mask()
        elif args.mode == "generate_fisher":
            runner = Diffusion(args, config)
            runner.generate_fisher()
    except Exception:
        logging.error(traceback.format_exc())

    return 0


if __name__ == "__main__":
    sys.exit(main())
