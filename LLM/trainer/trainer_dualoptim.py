from typing import Dict, Union, Any
from collections import OrderedDict

import torch
import copy
import importlib.metadata
import deepspeed
import datasets
from torch import nn
from transformers import Trainer
from transformers.utils import is_datasets_available
from transformers.trainer_utils import seed_worker
from torch.utils.data import ConcatDataset, DataLoader
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from .losses import get_loss

from trainer.custom_optimizer import AdamWDecouple8bit, AdamWDecoupleNormal
from trainer.custom_sampler import AlternatingSampler, DistributedAlternatingSampler


_smdistributed_available = importlib.util.find_spec("smdistributed") is not None


class CustomTrainerForgettingAlternate(Trainer):
    def __init__(
        self, alternate=True, optim_cfg="dual_adam", forget_lr=1e-5, retain_lr=1e-5, forget_freq=1, retain_freq=1,
        svd_rank=64, proj_update_freq=1, gp_forget=True, gp_retain=True, untar_to_tar_milestone=None,
        alpha=1., beta1=0.9, beta2=0.95, base_beta1=0.95, base_beta2=0.999,
        *args, **kwargs
    ):
        self.loss_type = kwargs.pop("loss_type")
        self.ref_model = kwargs.pop("ref_model")
        self.forget_coeff = kwargs.pop("forget_coeff")
        self.regularization_coeff = kwargs.pop("regularization_coeff")
        self.beta = kwargs.pop("beta")

        # Optimizer type (adam, sgd, dual_adam, dual_sgd)
        self.optim_cfg = optim_cfg
        self.forget_lr = forget_lr
        self.retain_lr = retain_lr
        self.alternate = alternate

        self.forget_lr_ratio = self.forget_lr / self.retain_lr

        self.forget_freq = forget_freq
        self.retain_freq = retain_freq
        self.step_count = 0

        # GP settings
        self.gp_forget = gp_forget
        self.gp_retain = gp_retain
        self.svd_rank = svd_rank
        self.proj_update_freq = proj_update_freq

        self.untar_to_tar_milestone = untar_to_tar_milestone

        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.base_beta1 = base_beta1
        self.base_beta2 = base_beta2

        super(CustomTrainerForgettingAlternate, self).__init__(*args, **kwargs)

        # Prepare the reference model with DeepSpeed
        if self.args.deepspeed is not None:
            self.ref_model = self.e_prepare_deepspeed(self.ref_model)
        if "full_shard" in self.args.fsdp:
            self.ref_model = FSDP(self.ref_model,
                                  sharding_strategy=FSDP.ShardingStrategy.FULL_SHARD,
                                  **self.args.fsdp_config)

    def _get_train_sampler(self) -> torch.utils.data.Sampler:
        """
        Override the default sampler to use AlternatingSampler
        """
        if not isinstance(self.train_dataset, ConcatDataset):
            raise ValueError("This CustomTrainer requires a ConcatDataset for train_dataset.")

        # train_dataset is a combination of dataset_a and dataset_b
        dataset_a, dataset_b = self.train_dataset.datasets

        return AlternatingSampler(
                dataset_a=dataset_a,
                dataset_b=dataset_b,
                batch_size=self.args.train_batch_size,
                m=self.forget_freq * self.args.gradient_accumulation_steps,
                n=self.retain_freq * self.args.gradient_accumulation_steps,
            )

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """
        Setup the optimizer and the learning rate scheduler.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method (or `create_optimizer` and/or
        `create_scheduler`) in a subclass.
        """
        opt_model = self.model
        decay_parameters = self.get_decay_parameter_names(opt_model)
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                ],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [
                    p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                ],
                "weight_decay": 0.,
            },
        ]

        if self.optim_cfg == "dual_adam":
            print("32bit DualAdam > Using forget ratio: ", self.forget_lr_ratio)
            print("Retain lr: ", self.retain_lr)

            self.optimizer = AdamWDecoupleNormal(
                optimizer_grouped_parameters,
                lr=self.retain_lr,
                lr_ratio_1=self.forget_lr_ratio,
                switch_freq_1=self.forget_freq,
                switch_freq_2=self.retain_freq,
                reinit_step=self.untar_to_tar_milestone
            )
        elif self.optim_cfg == "dual_adam_8bit":
            print("8bit DualAdam > Using forget ratio: ", self.forget_lr_ratio)
            print("Retain lr: ", self.retain_lr)

            self.optimizer = AdamWDecouple8bit(
                optimizer_grouped_parameters,
                lr=self.retain_lr,
                lr_ratio_1=self.forget_lr_ratio,
                switch_freq_1=self.forget_freq,
                switch_freq_2=self.retain_freq,
                reinit_step=self.untar_to_tar_milestone
            )
            import bitsandbytes

            manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

            skipped = 0
            for module in opt_model.modules():
                if isinstance(module, nn.Embedding):
                    skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                    # logger.info(f"skipped {module}: {skipped / 2 ** 20}M params")
                    manager.register_module_override(module, "weight", {"optim_bits": 32})
                    # logger.debug(f"bitsandbytes: will optimize {module} in fp32")
            print(f"skipped: {skipped / 2 ** 20}M params")
        else:
            self.create_optimizer()

        optimizer = self.optimizer
        self.create_scheduler(
            num_training_steps=num_training_steps, optimizer=optimizer
        )

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if self.alternate:  # alternating update
            if inputs[0] is not None or inputs[2] is not None:
                type = "forget"
            else:
                type = "retain"

            if type == "forget":
                forget_type = self.loss_type.split("+")[0]
                # print(forget_type)
                forget_loss, regularization_loss = get_loss(
                    model, self.ref_model, inputs, forget_type, self.beta
                )
                loss = self.forget_coeff * forget_loss

            else:
                retain_type = self.loss_type.split("+")[1]
                # print(retain_type)
                forget_loss, regularization_loss = get_loss(
                    model, self.ref_model, inputs, retain_type, self.beta
                )
                loss = self.regularization_coeff * regularization_loss

        else:  # joint update
            forget_loss, regularization_loss = get_loss(
                model, self.ref_model, inputs, self.loss_type, self.beta
            )

            loss = (
                self.forget_coeff * forget_loss
                + self.regularization_coeff * regularization_loss
            )

        return (loss, None) if return_outputs else loss

    def get_original_parameter_shape(self):
        param_shape_dict = OrderedDict()
        if self.args.deepspeed is not None:
            raise ValueError("DeepSpeed is not supported")
        if self.args.fsdp in [""]:
            for name, param in self.model.named_parameters():
                param_shape_dict[name] = param.shape
        else:
            with FSDP.summon_full_params(self.model, writeback=False):
                for name, param in self.model.named_parameters():
                    param_shape_dict[name] = param.shape

        return param_shape_dict

    def e_prepare_deepspeed(self, model):
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        config_kwargs = copy.deepcopy(deepspeed_plugin.deepspeed_config)
        # print(config_kwargs)

        if model is not None:
            if hasattr(model, "config"):
                hidden_size = (
                    max(model.config.hidden_sizes)
                    if getattr(model.config, "hidden_sizes", None)
                    else getattr(model.config, "hidden_size", None)
                )
                if (
                    hidden_size is not None
                    and config_kwargs["zero_optimization"]["stage"] == 3
                ):
                    config_kwargs.update(
                        {
                            "zero_optimization.reduce_bucket_size": hidden_size
                            * hidden_size,
                            "zero_optimization.stage3_param_persistence_threshold": 10
                            * hidden_size,
                            "zero_optimization.stage3_prefetch_bucket_size": 0.9
                            * hidden_size
                            * hidden_size,
                        }
                    )

        # If ZeRO-3 is used, we shard both the active and reference model.
        if config_kwargs["zero_optimization"]["stage"] != 3:
            config_kwargs["zero_optimization"]["stage"] = 0

        # Disable optimizer in DeepSpeed since we are using custom optimizers
        config_kwargs["optimizer"] = {"type": None}

        model, *_ = deepspeed.initialize(model=model, config=config_kwargs)
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

        return model


