import copy
import os
from collections import OrderedDict

import numpy as np

import arg_parser
import evaluation
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import unlearn
import utils
from trainer import validate


def main():
    args = arg_parser.parse_args()

    if torch.cuda.is_available():
        torch.cuda.set_device(int(args.gpu))
        device = torch.device(f"cuda:{int(args.gpu)}")
    else:
        device = torch.device("cpu")

    os.makedirs(args.save_dir, exist_ok=True)
    if args.seed:
        utils.setup_seed(args.seed)
    seed = args.seed
    # prepare dataset
    (
        model,
        train_loader_full,
        val_loader,
        test_loader,
        marked_loader,
    ) = utils.setup_model_dataset(args)
    model.cuda()

    def replace_loader_dataset(
        dataset, batch_size=args.batch_size, seed=1, shuffle=True
    ):
        utils.setup_seed(seed)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=0,
            pin_memory=True,
            shuffle=shuffle,
        )

    forget_dataset = copy.deepcopy(marked_loader.dataset)
    if args.dataset == "svhn":
        try:
            marked = forget_dataset.targets < 0
        except:
            marked = forget_dataset.labels < 0
        forget_dataset.data = forget_dataset.data[marked]
        try:
            forget_dataset.targets = -forget_dataset.targets[marked] - 1
        except:
            forget_dataset.labels = -forget_dataset.labels[marked] - 1
        forget_loader = replace_loader_dataset(forget_dataset, seed=seed, shuffle=True)
        retain_dataset = copy.deepcopy(marked_loader.dataset)
        try:
            marked = retain_dataset.targets >= 0
        except:
            marked = retain_dataset.labels >= 0
        retain_dataset.data = retain_dataset.data[marked]
        try:
            retain_dataset.targets = retain_dataset.targets[marked]
        except:
            retain_dataset.labels = retain_dataset.labels[marked]
        retain_loader = replace_loader_dataset(retain_dataset, seed=seed, shuffle=True)
        assert len(forget_dataset) + len(retain_dataset) == len(
            train_loader_full.dataset
        )
    else:
        try:
            marked = forget_dataset.targets < 0
            forget_dataset.data = forget_dataset.data[marked]
            forget_dataset.targets = -forget_dataset.targets[marked] - 1
            forget_loader = replace_loader_dataset(
                forget_dataset, seed=seed, shuffle=True
            )
            retain_dataset = copy.deepcopy(marked_loader.dataset)
            marked = retain_dataset.targets >= 0
            retain_dataset.data = retain_dataset.data[marked]
            retain_dataset.targets = retain_dataset.targets[marked]
            retain_loader = replace_loader_dataset(
                retain_dataset, seed=seed, shuffle=True
            )
            assert len(forget_dataset) + len(retain_dataset) == len(
                train_loader_full.dataset
            )
        except:
            marked = forget_dataset.targets < 0
            forget_dataset.imgs = forget_dataset.imgs[marked]
            forget_dataset.targets = -forget_dataset.targets[marked] - 1
            forget_loader = replace_loader_dataset(
                forget_dataset, seed=seed, shuffle=True
            )
            retain_dataset = copy.deepcopy(marked_loader.dataset)
            marked = retain_dataset.targets >= 0
            retain_dataset.imgs = retain_dataset.imgs[marked]
            retain_dataset.targets = retain_dataset.targets[marked]
            retain_loader = replace_loader_dataset(
                retain_dataset, seed=seed, shuffle=True
            )
            assert len(forget_dataset) + len(retain_dataset) == len(
                train_loader_full.dataset
            )

    print(f"number of retain dataset {len(retain_dataset)}")
    print(f"number of forget dataset {len(forget_dataset)}")
    unlearn_data_loaders = OrderedDict(
        retain=retain_loader, forget=forget_loader, val=val_loader, test=test_loader
    )

    criterion = nn.CrossEntropyLoss()

    evaluation_result = None
    checkpoint = None

    if args.resume:
        checkpoint = unlearn.load_unlearn_checkpoint(model, device, args)

    if args.resume and checkpoint is not None:
        model, evaluation_result = checkpoint
    else:

        checkpoint = torch.load(
            args.model_path, map_location=device, weights_only=False
        )
        if "state_dict" in checkpoint.keys():
            checkpoint = checkpoint["state_dict"]

        if args.mask_path:
            mask = torch.load(args.mask_path, map_location=device)
        else:
            mask = None

        if (
            args.unlearn != "retrain"
            and args.unlearn != "retrain_sam"
            and args.unlearn != "retrain_adv"
        ):
            model.load_state_dict(checkpoint, strict=False)

        # unlearn_method = unlearn.get_unlearn_method(args.unlearn)
        # unlearn_method(unlearn_data_loaders, model, criterion, args, mask)
        # unlearn.save_unlearn_checkpoint(model, None, args)

        loss_function = nn.CrossEntropyLoss()
        unlearn_method = unlearn.get_unlearn_method(args.unlearn)(
            model, loss_function, args.save_dir, args
        )
        unlearn_method.prepare_unlearn(unlearn_data_loaders)
        model, grad_var_forget, grad_var_retain, grad_mean_forget, grad_mean_retain, retain_loss, forget_loss, momentum_norm1, momentum_norm2, grad_norm_f, grad_norm_r, similarity_f, similarity_r, grad_sim, grad_sim1, momentum_sim, momentum_sim1, sim_fr, momentum_fr_sim = (
            unlearn_method.get_unlearned_model()
        )
        if len(grad_var_forget) > 0:
            np.save(
                os.path.join(args.save_dir, "grad_var_forget.npy"),
                np.array(grad_var_forget),
            )
        if len(grad_var_retain) > 0:
            np.save(
                os.path.join(args.save_dir, "grad_var_retain.npy"),
                np.array(grad_var_retain),
            )
        if len(grad_mean_forget) > 0:
            np.save(
                os.path.join(args.save_dir, "grad_mean_forget.npy"),
                np.array(grad_mean_forget),
            )
        if len(grad_mean_retain) > 0:
            np.save(
                os.path.join(args.save_dir, "grad_mean_retain.npy"),
                np.array(grad_mean_retain),
            )
        if len(retain_loss) > 0:
            np.save(
                os.path.join(args.save_dir, "retain_loss.npy"), np.array(retain_loss)
            )
        if len(forget_loss) > 0:
            np.save(
                os.path.join(args.save_dir, "forget_loss.npy"), np.array(forget_loss)
            )
        if len(momentum_norm1) > 0:
            np.save(
                os.path.join(args.save_dir, "momentum_norm1.npy"),
                np.array(momentum_norm1),
            )
        if len(momentum_norm2) > 0:
            np.save(
                os.path.join(args.save_dir, "momentum_norm2.npy"),
                np.array(momentum_norm2),
            )
        if len(grad_norm_f) > 0:
            np.save(
                os.path.join(args.save_dir, "grad_norm_f.npy"), np.array(grad_norm_f)
            )
        if len(grad_norm_r) > 0:
            np.save(
                os.path.join(args.save_dir, "grad_norm_r.npy"), np.array(grad_norm_r)
            )
        if len(similarity_f) > 0:
            np.save(
                os.path.join(args.save_dir, "similarity_f.npy"), np.array(similarity_f)
            )
        if len(similarity_r) > 0:
            np.save(
                os.path.join(args.save_dir, "similarity_r.npy"), np.array(similarity_r)
            )
        if len(grad_sim) > 0:
            np.save(os.path.join(args.save_dir, "grad_sim.npy"), np.array(grad_sim))
        if len(grad_sim1) > 0:
            np.save(os.path.join(args.save_dir, "grad_sim1.npy"), np.array(grad_sim1))
        if len(momentum_sim) > 0:
            np.save(
                os.path.join(args.save_dir, "momentum_sim.npy"), np.array(momentum_sim)
            )
        if len(momentum_sim1) > 0:
            np.save(
                os.path.join(args.save_dir, "momentum_sim1.npy"),
                np.array(momentum_sim1),
            )
        if len(sim_fr) > 0:
            np.save(os.path.join(args.save_dir, "sim_fr.npy"), np.array(sim_fr))
        if len(momentum_fr_sim) > 0:
            np.save(
                os.path.join(args.save_dir, "momentum_fr_sim.npy"),
                np.array(momentum_fr_sim),
            )

    if evaluation_result is None:
        evaluation_result = {}

    if "new_accuracy" not in evaluation_result:
        accuracy = {}
        for name, loader in unlearn_data_loaders.items():
            if name == "val":
                continue
            utils.dataset_convert_to_test(loader.dataset, args)
            val_acc = validate(loader, model, criterion, args)
            accuracy[name] = val_acc
            print(f"{name} acc: {val_acc if name != 'forget' else 100-val_acc}")

        evaluation_result["accuracy"] = accuracy
        unlearn.save_unlearn_checkpoint(model, evaluation_result, args)

    for deprecated in ["MIA", "SVC_MIA", "SVC_MIA_forget"]:
        if deprecated in evaluation_result:
            evaluation_result.pop(deprecated)

    """forget efficacy MIA:
        in distribution: retain
        out of distribution: test
        target: (, forget)"""
    utils.dataset_convert_to_test(retain_dataset, args)
    utils.dataset_convert_to_test(forget_loader, args)
    utils.dataset_convert_to_test(test_loader, args)
    # if "SVC_MIA_forget_efficacy" not in evaluation_result:
    #     test_len = len(test_loader.dataset)
    #
    #     shadow_train = torch.utils.data.Subset(retain_dataset, list(range(test_len)))
    #     shadow_train_loader = torch.utils.data.DataLoader(
    #         shadow_train, batch_size=args.batch_size, shuffle=False
    #     )
    #
    #     evaluation_result["SVC_MIA_forget_efficacy"] = evaluation.SVC_MIA(
    #         shadow_train=shadow_train_loader,
    #         shadow_test=test_loader,
    #         target_train=None,
    #         target_test=forget_loader,
    #         model=model,
    #     )
    #     unlearn.save_unlearn_checkpoint(model, evaluation_result, args)

    for mia_metric in ["entropy"]:
        # if f"{mia_metric} mia" not in evaluation_result:
        evaluation_result[f"{mia_metric} mia"] = evaluation.get_membership_attack_prob(
            retain_loader, forget_loader, test_loader, model, mia_metric
        )

    # print(f"MIA: {evaluation_result['SVC_MIA_forget_efficacy']['confidence']}")
    print(f"MIA: {evaluation_result[f'{mia_metric} mia']}")
    unlearn.save_unlearn_checkpoint(model, evaluation_result, args)


if __name__ == "__main__":
    main()
