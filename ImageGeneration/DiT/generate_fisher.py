import torch

# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.utils import save_image
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
from tqdm import tqdm
import argparse
import logging
import os

from models import DiT_models
from download import find_model
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from unlearn_dataset import get_unlearn_dataset

import warnings

warnings.filterwarnings("ignore")
#################################################################################
#                             Training Helper Functions                         #
#################################################################################


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """

    logging.basicConfig(
        level=logging.INFO,
        format="[\033[34m%(asctime)s\033[0m] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f"{logging_dir}/log.txt"),
        ],
    )
    logger = logging.getLogger(__name__)

    return logger


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(
        arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]
    )


def cycle(dl):
    while True:
        for data in dl:
            yield data


@torch.no_grad()
def sample_visualization(
    model,
    diffusion,
    vae,
    forward_with_cfg,
    latent_size,
    device,
    train_steps,
    checkpoint_dir,
):
    print(f"Visualization at {train_steps:07d}")
    model.eval()  # important! This disables randomized embedding dropout
    # Labels to condition the model with (feel free to change):
    class_labels = [207, 360, 387, 972, 89]
    # 979, 417, 279, 270, 980,
    # 250, 291, 388, 33, 812]

    # Create sampling noise:
    n = len(class_labels)
    z = torch.randn(n, 4, latent_size, latent_size, device=device)
    y = torch.tensor(class_labels, device=device)

    # Setup classifier-free guidance:
    z = torch.cat([z, z], 0)
    y_null = torch.tensor([1000] * n, device=device)
    y = torch.cat([y, y_null], 0)
    model_kwargs = dict(y=y, cfg_scale=4.0)

    # Sample images:
    samples = diffusion.p_sample_loop(
        forward_with_cfg,
        z.shape,
        z,
        clip_denoised=False,
        model_kwargs=model_kwargs,
        progress=True,
        device=device,
    )
    samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
    samples = vae.decode(samples / 0.18215).sample

    # Save and display images:
    image_save_path = os.path.join(checkpoint_dir, f"{train_steps:07d}_sample.png")
    save_image(samples, image_save_path, nrow=5, normalize=True, value_range=(-1, 1))
    model.train()


#################################################################################
#                                  Training Loop                                #
#################################################################################


def main(args):
    """
    Trains a new DiT model.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DP:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = args.seed
    torch.manual_seed(seed)
    print(f"Starting seed={seed}")

    # Setup an experiment folder:
    os.makedirs(
        args.results_dir, exist_ok=True
    )  # Make results folder (holds all experiment subfolders)
    experiment_index = len(glob(f"{args.results_dir}/*"))
    model_string_name = args.model.replace(
        "/", "-"
    )  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
    experiment_dir = (
        f"{args.results_dir}/{experiment_index:03d}-{model_string_name}-fisher"
    )  # Create an experiment folder
    checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
    os.makedirs(checkpoint_dir, exist_ok=True)
    logger = create_logger(experiment_dir)
    logger.info(f"Experiment directory created at {experiment_dir}")

    # Create model:
    assert (
        args.image_size % 8 == 0
    ), "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
    model = DiT_models[args.model](input_size=latent_size, num_classes=args.num_classes)
    forward_with_cfg = model.forward_with_cfg
    # load from ckpt
    if args.ckpt:
        print(f"Load pretrain from {args.ckpt}")
        ckpt_path = args.ckpt
        state_dict = find_model(ckpt_path)
        model.load_state_dict(state_dict)

    # Note that parameter initialization is done within the DiT constructor
    ema = deepcopy(model).to(
        device
    )  # Create an EMA of the model for use after training
    requires_grad(ema, False)
    model.to(device)
    model = torch.nn.DataParallel(model)
    diffusion = create_diffusion(
        timestep_respacing=""
    )  # default: 1000 steps, linear noise schedule
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    opt = torch.optim.AdamW(model.parameters(), lr=0, weight_decay=0)

    # Setup data:
    transform = transforms.Compose(
        [
            transforms.Lambda(
                lambda pil_image: center_crop_arr(pil_image, args.image_size)
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True
            ),
        ]
    )
    forget_dataset, remain_dataset = get_unlearn_dataset(
        args.data_path, args.forget_class, transform
    )
    forget_loader = DataLoader(
        forget_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    remain_loader = DataLoader(
        remain_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    forget_iter = cycle(forget_loader)
    remain_iter = cycle(remain_loader)
    logger.info(
        f"Forget Dataset contains {len(forget_dataset):,} images ({args.data_path})"
    )
    logger.info(
        f"Remain Dataset contains {len(remain_dataset):,} images ({args.data_path})"
    )

    # Prepare models for fisher:
    model.eval()
    os.makedirs(os.path.join(args.mask_path, str(args.forget_class)), exist_ok=True)
    logger.info(f"Save in {os.path.join(args.mask_path, str(args.forget_class))}")
    start_time = time()
    train_steps = 0
    log_steps = 0

    logger.info(f"Getting forget fisher...")
    forget_gradients = {}
    for name, param in model.named_parameters():
        forget_gradients[name] = 0
    for step in tqdm(
        range(0, args.n_iters), total=args.n_iters, desc="Generate forget fisher"
    ):
        # forget stage
        forget_x, forget_y = next(forget_iter)
        forget_x = forget_x.to(device)
        forget_y = forget_y.to(device)
        with torch.no_grad():
            # Map input images to latent space + normalize latents:
            forget_x = vae.encode(forget_x).latent_dist.sample().mul_(0.18215)
        forget_t = torch.randint(
            0, diffusion.num_timesteps, (forget_x.shape[0],), device=device
        )

        forget_model_kwargs = dict(y=forget_y)
        forget_loss_dict = diffusion.training_losses(
            model, forget_x, forget_t, forget_model_kwargs
        )
        forget_loss = forget_loss_dict["loss"].mean()

        opt.zero_grad()
        forget_loss.backward()
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.grad is not None:
                    forget_gradients[name] += (
                        param.grad.data.cpu() ** 2
                    ) / args.n_iters

        log_steps += 1
        train_steps += 1
        if train_steps % args.log_every == 0:
            # Measure training speed:
            end_time = time()
            steps_per_sec = log_steps / (end_time - start_time)
            logger.info(
                f"(step={train_steps:07d}) Train Steps/Sec: {steps_per_sec:.2f}"
            )
            log_steps = 0
            start_time = time()

    torch.save(
        forget_gradients,
        os.path.join(args.mask_path, str(args.forget_class), "forget_fisher.pt"),
    )
    ###############
    logger.info(f"Getting remain fisher...")
    model.eval()
    start_time = time()
    train_steps = 0
    log_steps = 0

    remain_gradients = {}
    for name, param in model.named_parameters():
        remain_gradients[name] = 0
    for step in tqdm(
        range(0, args.n_iters), total=args.n_iters, desc="Generate remain fisher"
    ):
        # remain stage
        remain_x, remain_y = next(remain_iter)
        remain_x = remain_x.to(device)
        remain_y = remain_y.to(device)
        with torch.no_grad():
            # Map input images to latent space + normalize latents:
            remain_x = vae.encode(remain_x).latent_dist.sample().mul_(0.18215)
        remain_t = torch.randint(
            0, diffusion.num_timesteps, (remain_x.shape[0],), device=device
        )
        remain_model_kwargs = dict(y=remain_y)
        remain_loss_dict = diffusion.training_losses(
            model, remain_x, remain_t, remain_model_kwargs
        )
        remain_loss = remain_loss_dict["loss"].mean()
        opt.zero_grad()
        remain_loss.backward()
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.grad is not None:
                    remain_gradients[name] += (
                        param.grad.data.cpu() ** 2
                    ) / args.n_iters

        log_steps += 1
        train_steps += 1
        if train_steps % args.log_every == 0:
            # Measure training speed:
            end_time = time()
            steps_per_sec = log_steps / (end_time - start_time)
            logger.info(
                f"(step={train_steps:07d}) Train Steps/Sec: {steps_per_sec:.2f}"
            )
            log_steps = 0
            start_time = time()

    torch.save(
        remain_gradients,
        os.path.join(args.mask_path, str(args.forget_class), "remain_fisher.pt"),
    )

    logger.info("Done!")


if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument(
        "--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2"
    )
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--n-iters", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--vae", type=str, choices=["ema", "mse"], default="ema"
    )  # Choice doesn't affect training
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).",
    )
    # unlearn
    parser.add_argument(
        "--forget-class", type=int, required=True, nargs="?", help="class to forget"
    )
    parser.add_argument(
        "--mask-path", type=str, default=None, help="the path to store mask"
    )
    parser.add_argument("--gpu", type=str, default="0")
    args = parser.parse_args()
    main(args)
