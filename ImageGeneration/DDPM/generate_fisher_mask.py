import os
import argparse

import torch


def calc_sparsity(tensor):
    # Count zero elements
    num_zero_elements = tensor.numel() - torch.count_nonzero(tensor)

    # Total number of elements
    total_elements = tensor.numel()

    # Compute sparsity
    sparsity = num_zero_elements / total_elements
    return sparsity.item(), total_elements, num_zero_elements


parser = argparse.ArgumentParser()
parser.add_argument(
    "--ckpt_folder", type=str, required=True, help="Path to fisher ckpt path"
)
parser.add_argument(
    "--threshold",
    type=float,
    default=1.0,
    help="Saliency map threshold, lambda in paper",
)

args = parser.parse_args()

ckpt_path = args.ckpt_folder
ff = os.path.join(ckpt_path, "forget_fisher.pt")
rf = os.path.join(ckpt_path, "remain_fisher.pt")

forget_fisher = torch.load(ff)
remain_fisher = torch.load(rf)

th = args.threshold

total_cnt = 0
w_cnt = 0
gradients = {}
for name, m in forget_fisher.items():
    weight_saliency = (forget_fisher[name] + 1e-15) / (remain_fisher[name] + 1e-15)
    w = weight_saliency >= th
    w_sparsity, total_elements, w_num_zero_elements = calc_sparsity(w)
    w_cnt += w_num_zero_elements
    total_cnt += total_elements
    gradients[name] = w

print(f"Total sparsity th:{th} weight:{w_cnt/total_cnt*100}")
torch.save(gradients, os.path.join(ckpt_path, f"fisher_{th}.pt"))
