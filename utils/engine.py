import torch
import numpy as np
from torch.utils.data import DataLoader
from datasets.rppg_datasets import VIPL, UBFC, PURE, COHFACE, BUAA, MMPD, MMSE
from losses.NPLoss import Neg_Pearson
from losses.CELoss import CrossEntropyKL
from archs.ViT import ViT

dataset_path_map = {
    "VIPL": "PATH_TO_VIPL",
    "UBFC": "PATH_TO_UBFC",
    "PURE": "PATH_TO_PURE",
    "BUAA": "PATH_TO_BUAA",
    "MMPD": "PATH_TO_MMPD",
}


def _init_fn(seed=92):
    np.random.seed(seed)


def build_one_dataset(dataset_name, args, mode):
    mode_num_rppg = {
        "train": args.num_rppg,
        "train_all": args.num_rppg,
        "test_all": -1,
        "test": -1,
    }

    num_rppg = mode_num_rppg[mode]
    train_mode = mode

    if dataset_name == "VIPL":
        dataset = VIPL(
            data_dir=dataset_path_map["VIPL"],
            T=num_rppg,
            train=train_mode,
            w=args.img_size,
            h=args.img_size,
            fold=args.vipl_fold,
        )
    elif dataset_name == "VIPL1to4":
        # fold_list=[1,2,3,4], fold=1 is test, 2-4 are train
        dataset = VIPL(
            data_dir=dataset_path_map["VIPL"],
            T=num_rppg,
            train=train_mode,
            w=args.img_size,
            h=args.img_size,
            fold_list=[1, 2, 3, 4],
        )
    elif dataset_name == "VIPL5":
        # fold=5, internally divide train/test again
        dataset = VIPL(
            data_dir=dataset_path_map["VIPL"],
            T=num_rppg,
            train=train_mode,
            w=args.img_size,
            h=args.img_size,
            fold_list=[5],
            split_ratio=args.vipl5_split_ratio,
        )
    # -------------------------------------------------------
    ## UBFC dataset
    elif dataset_name == "UBFC":
        dataset = UBFC(
            data_dir=dataset_path_map[dataset_name],
            T=num_rppg,
            train=train_mode,
            w=args.img_size,
            h=args.img_size,
        )
    # -------------------------------------------------------
    ## PURE dataset
    elif dataset_name == "PURE":
        dataset = PURE(
            data_dir=dataset_path_map[dataset_name],
            T=num_rppg,
            train=train_mode,
            w=args.img_size,
            h=args.img_size,
        )
    # -------------------------------------------------------
    ## COHFACE dataset
    elif dataset_name == "COHFACE":
        dataset = COHFACE(
            data_dir=dataset_path_map[dataset_name],
            T=num_rppg,
            train=train_mode,
            w=args.img_size,
            h=args.img_size,
        )
    # --------------------------------------------------------
    ## BUAA dataset
    elif dataset_name == "BUAA":
        dataset = BUAA(
            data_dir=dataset_path_map[dataset_name],
            T=num_rppg,
            train=train_mode,
            w=args.img_size,
            h=args.img_size,
        )
    elif dataset_name == "MMPD":
        dataset = MMPD(
            data_dir=dataset_path_map[dataset_name],
            T=num_rppg,
            train=train_mode,
            w=args.img_size,
            h=args.img_size,
        )
    elif dataset_name == "MMSE":
        dataset = MMSE(
            data_dir=dataset_path_map[dataset_name],
            T=num_rppg,
            train=train_mode,
            w=args.img_size,
            h=args.img_size,
        )
    else:
        raise NotImplementedError

    return dataset


def build_dataset(args, mode, batch_size):
    shuffle_flag = mode == "train"

    datasets = args.dataset.split("_")

    all_dataloader = []

    for dataset_name in datasets:
        dataset = build_one_dataset(dataset_name, args, mode)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=4,
            worker_init_fn=_init_fn,
        )
        all_dataloader.append(dataloader)

    return all_dataloader


def build_optimizer(args, model):
    # Check if there are adapter weight groups
    param_groups = []
    # Adapt ViT multi-layer adapter weights
    adapter_weight_lr_new = getattr(args, "adapter_weight_lr_new", 1e-3)
    adapter_weight_lr_old = getattr(args, "adapter_weight_lr_old", 1e-4)
    # Only support ViT architecture
    if hasattr(model, "adapter_list_1"):
        for adapter_list in [
            model.adapter_list_1,
            model.adapter_list_2,
            model.adapter_list_3,
        ]:
            num_adapters = len(adapter_list.adapter_weights)
            if num_adapters > 0:
                # New weights (last one)
                param_groups.append(
                    {
                        "params": [adapter_list.adapter_weights[-1]],
                        "lr": adapter_weight_lr_new,
                    }
                )
                # Old weights (previous ones)
                if num_adapters > 1:
                    param_groups.append(
                        {
                            "params": [w for w in adapter_list.adapter_weights[:-1]],
                            "lr": adapter_weight_lr_old,
                        }
                    )
        # Other parameters (backbone, adapter, vae, etc.), exclude all adapter_weights
        adapter_weight_ids = set()
        for adapter_list in [
            model.adapter_list_1,
            model.adapter_list_2,
            model.adapter_list_3,
        ]:
            for w in adapter_list.adapter_weights:
                adapter_weight_ids.add(id(w))

        other_params = []
        for name, param in model.named_parameters():
            if id(param) not in adapter_weight_ids:
                other_params.append(param)
        param_groups.append({"params": other_params, "lr": args.lr})
        if args.optim == "adam":
            optimizer = torch.optim.Adam(
                param_groups,
                betas=(args.beta1, args.beta2),
                weight_decay=args.weight_decay,
            )
        elif args.optim == "sgd":
            optimizer = torch.optim.SGD(
                param_groups,
                momentum=args.momentum,
                weight_decay=args.weight_decay,
            )
        else:
            raise NotImplementedError
        return optimizer
    else:
        # Non-ViT model, keep original logic
        if args.optim == "adam":
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=args.lr,
                betas=(args.beta1, args.beta2),
                weight_decay=args.weight_decay,
            )
        elif args.optim == "sgd":
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=args.lr,
                momentum=args.momentum,
                weight_decay=args.weight_decay,
            )
        else:
            raise NotImplementedError
        return optimizer


def build_scheduler(args, optimizer):
    if args.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=args.step_size,
            gamma=args.gamma,
        )
    elif args.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=args.factor,
            patience=args.patience,
            verbose=True,
            threshold=args.threshold,
            threshold_mode="rel",
            cooldown=0,
            min_lr=args.min_lr,
            eps=args.eps,
        )
    elif args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.T_max,
            eta_min=args.eta_min,
        )
    else:
        raise NotImplementedError
    return scheduler


def build_criterion(args):
    from losses.NPLoss import Neg_Pearson  # Correct import name
    from losses.CELoss import CrossEntropyKL
    from losses.VAELoss import VAELoss  # Add VAELoss import

    criterion = {}

    # Extract loss function list from args.loss
    loss_list = eval(args.loss)

    for loss_name in loss_list:
        if loss_name == "np_loss":
            criterion["np_loss"] = Neg_Pearson()  # Use correct class name
        elif loss_name == "ce_loss":
            criterion["ce_loss"] = CrossEntropyKL()
        elif loss_name == "vae_loss":  # Add VAELoss construction
            vae_kl_weight = getattr(args, "vae_kl_weight", 0.001)
            criterion["vae_loss"] = VAELoss(kl_weight=vae_kl_weight)
        else:
            raise ValueError(f"Unknown loss function: {loss_name}")

    return criterion


def build_model(args) -> torch.nn.Module:
    if args.model == "ViT":
        # Read adapter related parameters from args, prioritize command line/config values
        initial_adapters = getattr(args, "initial_adapters", 1)
        adapter_threshold = getattr(args, "adapter_threshold", 0.05)
        backbone_weight = getattr(args, "backbone_weight", 0.9)
        adapter_weight = getattr(args, "adapter_weight", 0.1)
        model = ViT(
            image_size=(args.num_rppg, args.img_size, args.img_size),
            patches=(4, 4, 4),
            dim=64,
            ff_dim=144,
            num_heads=4,
            num_layers=12,
            dropout_rate=0.1,
            initial_adapters=initial_adapters,
            adapter_threshold=adapter_threshold,
            backbone_weight=backbone_weight,
            adapter_weight=adapter_weight,
        )  # First domain initializes 1 adapter
    else:
        raise NotImplementedError
    return model
