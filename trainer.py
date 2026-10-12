import torch
import os
import numpy as np
import random
import time
import argparse
import logging
import shutil
import torch.nn.functional as F

from copy import deepcopy
from tqdm import tqdm

from utils.engine import (
    build_dataset,
    build_optimizer,
    build_scheduler,
    build_criterion,
    build_model,
    _init_fn,
)
from torch.utils.data import SubsetRandomSampler, DataLoader
from utils.util import (
    AvgrageMeter,
    pearson_correlation_coefficient,
    update_avg_meters,
    cal_psd_hr,
)


def set_seed(seed=92):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


class RppgEstimatorTrainer:
    # progressive_reg distribution alignment parameters
    def __init__(self, args) -> None:
        self.args = args
        self.progressive_reg_mode = getattr(
            args, "progressive_reg_mode", "swd"
        )  # 'mse' or 'swd'
        self.progressive_swd_K = int(
            getattr(args, "progressive_swd_K", 16)
        )  # SWD direction number
        self.progressive_reg_weight = float(
            getattr(args, "progressive_reg_weight", 1.0)
        )  # regularization weight

        self.gpu_list = [int(i) for i in args.gpu.split(",")]
        self.gpu_num = len(self.gpu_list)
        self.batch_size = args.batch_size
        self.device = torch.device(f"cuda:{self.gpu_list[0]}")

        self.rppg_estimator = build_model(args).to(self.device)

        ## generate save path
        self.run_date = time.strftime("%m%d_%H%M", time.localtime(time.time()))
        self.save_path = f"{args.save_path}/{self.run_date}"

        ## dataloader NOTE: SELECT YOUR DATASET
        self.all_datasets = self.args.dataset.split(
            "_"
        )  # 'MMPD_VIPLF5_BUAA_UBFC_PURE_VIPLF14_COHFACE' => ['MMPD', 'VIPLF5', 'BUAA', 'UBFC', 'PURE', 'VIPLF14', 'COHFACE']
        initial_epoch, cl_epoch = [int(i) for i in self.args.epochs.split("_")]
        self.all_epochs = [initial_epoch] + [
            cl_epoch for _ in range(len(self.all_datasets) - 1)
        ]
        self.train_dataloader = build_dataset(
            args, mode="train", batch_size=self.batch_size
        )
        self.val_dataloader = build_dataset(args, mode="test", batch_size=1)

        # Gracefully handle loss functions and weights, automatically add vae_loss
        loss_list = eval(args.loss)
        weight_list = eval(args.loss_weight)
        if getattr(args, "use_vae_loss", False) and "vae_loss" not in loss_list:
            loss_list.append("vae_loss")
            weight_list.append(0.1)
        if len(weight_list) < len(loss_list):
            weight_list += [1.0] * (len(loss_list) - len(weight_list))
        args.loss = str(loss_list)
        args.loss_weight = str(weight_list)

        ## optimizer
        self.optimizer = build_optimizer(args, self.rppg_estimator)
        self.scheduler = build_scheduler(args, self.optimizer)
        self.loss_funcs = build_criterion(args)
        self.loss_funcs_weight = dict(zip(loss_list, weight_list))

        ## loss & metrics saver
        self.loss_meters = dict(
            [(key, AvgrageMeter()) for key in self.loss_funcs.keys()]
        )
        # Ensure progressive_reg also has a meter
        if "progressive_reg" not in self.loss_meters:
            self.loss_meters["progressive_reg"] = AvgrageMeter()
        self.metrics_meters = {
            "mae": AvgrageMeter(),
        }

        ## constant
        self.bpm_range = torch.arange(40, 180, dtype=torch.float).to(self.device)
        num_datasets = len(self.all_datasets)
        self.best_epoch = [0] * num_datasets
        self.best_val_mae = [1000] * num_datasets  # mean absolute error
        self.best_val_rmse = [1000] * num_datasets  # root mean square error
        self.best_sd = [1000] * num_datasets  # standard deviation
        self.best_r = [0] * num_datasets  # Pearson's correlation coefficient
        self.frame_rate = 30

        # Initialize last epoch metrics
        self.last_val_mae = [1000] * num_datasets
        self.last_val_rmse = [1000] * num_datasets
        self.last_sd = [1000] * num_datasets
        self.last_r = [0] * num_datasets

        # Add intelligent expansion related parameters
        self.is_first_domain = True
        self.vae_training_epochs = getattr(
            args, "vae_training_epochs", 5
        )  # Number of epochs to train VAE
        self.adapter_threshold = getattr(
            args, "adapter_threshold", 0.5
        )  # Threshold for adding adapters

        # VAE distribution comparison parameters
        wasserstein_norm = getattr(args, "wasserstein_norm", "False")
        self.normalize_wasserstein = wasserstein_norm != "False"
        self.wasserstein_norm_method = (
            wasserstein_norm if wasserstein_norm != "False" else "l2"
        )

        # PGP constraint annealing rate (paper's kappa in exp(-kappa * s)).
        self.progressive_beta0 = float(getattr(args, "progressive_beta0", 1.0))
        self.progressive_beta_gamma = float(
            getattr(args, "progressive_beta_gamma", 1.0)
        )

        # If the model supports, set normalization parameters for Wasserstein distance calculation
        if hasattr(self.rppg_estimator, "set_wasserstein_params"):
            self.rppg_estimator.set_wasserstein_params(
                normalize=self.normalize_wasserstein,
                norm_method=self.wasserstein_norm_method,
            )

        # Save adapter configuration information for each domain
        self.domain_adapter_configs = (
            {}
        )  # Format: {domain_name: (layer1_count, layer2_count, layer3_count)}
        self.distribution_store_path = None

        # Initialize adapter configuration for the first domain
        first_domain = self.all_datasets[0]
        self.domain_adapter_configs[first_domain] = (
            1,
            1,
            1,
        )  # Initially 1 adapter per layer

        # ========== Sample Annealing (Offline Precomputation + Dynamic Sampler) Related State ==========
        # Enable switch: Only effective from the second domain onwards
        self.enable_sample_anneal = getattr(args, "sample_anneal", True)
        # Layer used (consistent with distribution comparison, default layer 2 pre-FFN features)
        self.sample_anneal_layer = int(getattr(args, "sample_anneal_layer", 2))
        # Initial coverage ratio q0 (first train samples closer to parent domain in top q0 quantile)
        self.sample_anneal_q0 = float(getattr(args, "sample_anneal_q0", 0.3))
        # Annealing window to full training (number of epochs); -1 means use 50% of total epochs for this domain
        self.sample_anneal_warmup_epochs = int(
            getattr(args, "sample_anneal_warmup_epochs", -1)
        )
        # Exploration ratio: additional random mix from full domain each epoch (prevent bias)
        self.sample_anneal_explore = float(getattr(args, "sample_anneal_explore", 0.1))
        # Scheme C (dynamic sample-anneal) parameters
        self.sample_w_ema_alpha = float(getattr(args, "sample_w_ema_alpha", 0.9))
        self.sample_w_sigmoid_s = float(getattr(args, "sample_w_sigmoid_s", 1.0))
        self.sample_w_sigmoid_z0 = float(getattr(args, "sample_w_sigmoid_z0", 0.0))
        self.sample_w_beta = float(getattr(args, "sample_w_beta", 0.5))
        # warmup epochs before enabling W-driven modulation (default 1)
        self.sample_warmup_epochs = int(getattr(args, "sample_warmup_epochs", 1))
        # if finite ratio (valid precompute) below this, fall back to linear schedule
        self.sample_finite_ratio_thresh = float(
            getattr(args, "sample_finite_ratio_thresh", 0.2)
        )
        # minimum core ratio
        self.sample_k_min_ratio = float(getattr(args, "sample_k_min_ratio", 0.01))
        # precomputed cache: {dataset_idx: { 'dist': np.ndarray, 'order': np.ndarray, 'N': int }}
        self.sample_curriculum = {}
        # per-dataset running stats for dynamic mapping (EMA)
        # structure: {dataset_idx: {'W_ema': float, 'med_ema': float, 'mad_ema': float}}
        self.sample_stats = {}

        # ========== Adapter output distribution reference buffer (previous round reference distribution Rt) ==========
        # Each domain maintains a previous round reference distribution (prev) and current round accumulation (cur_collect), stored separately by layer
        self.adapter_ref_prev = (
            {}
        )  # {dataset_idx: { 'x_attn_1': Tensor[N,D]_cpu, 'x_attn_2': ..., 'x_attn_3': ... }}
        self.adapter_ref_cur_collect = (
            {}
        )  # {dataset_idx: { 'x_attn_1': [Tensor[k,D]_cpu, ...], ... }}
        # Buffer capacity and sampling control
        self.adapter_ref_capacity = int(getattr(args, "adapter_ref_capacity", 8192))
        self.adapter_ref_init_batches = int(
            getattr(args, "adapter_ref_init_batches", 5)
        )
        self.adapter_ref_tokens_per_batch = int(
            getattr(args, "adapter_ref_tokens_per_batch", 512)
        )
        self.expanded_adapter_layers = set()

    def intelligent_expand_adapters_for_new_domain(self, dataset_idx):
        """Smart expansion: decide whether to add adapter based on VAE distribution distance"""
        if not (
            hasattr(self.rppg_estimator, "intelligent_expand")
            and hasattr(self.rppg_estimator, "use_adapters")
            and self.rppg_estimator.use_adapters
        ):
            self.logger.info("Model does not support intelligent adapter expansion")
            return False

        # Get adapter info before expansion
        if hasattr(self.rppg_estimator, "get_adapter_info"):
            old_adapter_info = self.rppg_estimator.get_adapter_info()
        else:
            old_adapter_info = "No adapter info available"

        # Use small batch for intelligent expansion decision
        current_domain_name = self.all_datasets[dataset_idx]

        try:
            # Intelligent expansion
            expansion_info = self.rppg_estimator.intelligent_expand(
                self.train_dataloader[dataset_idx], new_domain_name=current_domain_name
            )
            self.expanded_adapter_layers = {
                f"x_attn_{layer_idx}"
                for layer_idx in (1, 2, 3)
                if str(expansion_info.get(f"layer_{layer_idx}", "")).startswith(
                    "Added adapter"
                )
            }

            # Get adapter info after expansion
            if hasattr(self.rppg_estimator, "get_adapter_info"):
                new_adapter_info = self.rppg_estimator.get_adapter_info()
            else:
                new_adapter_info = "No adapter info available"

            # Print adapter expansion information
            self.logger.info(f"Intelligent expansion for domain {current_domain_name}:")
            self.logger.info(f"Old adapter info: {old_adapter_info}")
            self.logger.info(f"Expansion decisions: {expansion_info}")
            self.logger.info(
                f'Layer 1 VAE distances: {expansion_info.get("layer_1_distances", [])}'
            )
            self.logger.info(
                f'Layer 2 VAE distances: {expansion_info.get("layer_2_distances", [])}'
            )
            self.logger.info(
                f'Layer 3 VAE distances: {expansion_info.get("layer_3_distances", [])}'
            )
            self.logger.info(f"New adapter info: {new_adapter_info}")

            # Rebuild optimizer to include new adapter parameters
            self.optimizer = build_optimizer(self.args, self.rppg_estimator)
            self.scheduler = build_scheduler(self.args, self.optimizer)

            # Ensure the expanded model is fully on the correct device
            self.rppg_estimator.to(self.device)

            # Update adapter configuration information
            current_domain = self.all_datasets[dataset_idx]
            current_config = (
                self.rppg_estimator.adapter_list_1.get_num_adapters(),
                self.rppg_estimator.adapter_list_2.get_num_adapters(),
                self.rppg_estimator.adapter_list_3.get_num_adapters(),
            )
            self.domain_adapter_configs[current_domain] = current_config
            self.logger.info(
                f"Updated adapter config for {current_domain}: {current_config}"
            )

            # Initialize new adapter feature storage
            try:
                self.rppg_estimator.initialize_new_adapters(
                    self.train_dataloader[dataset_idx]
                )
            except Exception as e:
                self.logger.warning(f"Failed to initialize new adapter features: {e}")
                print(f"Warning: Failed to initialize new adapter features: {e}")

            self.logger.info(
            self.logger.info("Rebuilt optimizer and scheduler for intelligently expanded adapters")
            )

            return True

        except Exception as e:
            self.logger.error(f"Error during intelligent expansion: {e}")
            return False

    # ================= Sample annealing: offline precomputation + dynamic Sampler =================
    def _get_parent_for_layer(self, layer_idx: int, dataset_idx: int):
        """Get parent domain name and parent adapter index for specified layer (fallback if missing).

        Returns: (parent_domain_name: str, parent_idx: int, adapter_list)
        """
        layer_key = f"layer_{layer_idx}"
        parent_domain = None
        parent_idx = None
        adapter_list = None
        if (
            hasattr(self.rppg_estimator, "parent_info")
            and self.rppg_estimator.parent_info
        ):
            info = self.rppg_estimator.parent_info.get(layer_key, None)
            if info is not None:
                parent_domain = info.get("domain", None)
                parent_idx = info.get("parent_idx", None)
        # Select corresponding layer's adapter_list
        adapter_attr = f"adapter_list_{layer_idx}"
        if hasattr(self.rppg_estimator, adapter_attr):
            adapter_list = getattr(self.rppg_estimator, adapter_attr)

        # Fallback strategy: if parent domain is missing, use the previous domain; if parent idx is missing, use 0
        if parent_domain is None:
            # Fallback to the previous domain name of current new domain (more consistent with time order)
            if hasattr(self, "all_datasets") and dataset_idx > 0:
                parent_domain = self.all_datasets[dataset_idx - 1]
        if parent_idx is None:
            parent_idx = 0

        return parent_domain, int(parent_idx), adapter_list

    @torch.no_grad()
    def precompute_sample_distances(self, dataset_idx: int):
        """Precompute the closeness of each sample to the parent domain for the new domain (Wasserstein distance, smaller is closer).

        - Use pre-FFN features of selected layer (default layer 2): outputs['x_attn_{layer}'].
        - Use VAE of parent adapter: adapter_list_{layer}.adapter_vaes[parent_idx].
        - Compare with saved distribution of parent domain parent_domain (needs to be saved before entering this domain).
        """
        if not self.enable_sample_anneal or dataset_idx == 0:
            return

        parent_domain, parent_idx, adapter_list = self._get_parent_for_layer(
            self.sample_anneal_layer, dataset_idx
        )
        if adapter_list is None or len(adapter_list.adapter_vaes) == 0:
            print(
                f"[SampleAnneal] Skip precompute: adapter/vae not ready for layer {self.sample_anneal_layer}"
            )
            return

        parent_vae = adapter_list.adapter_vaes[
            min(parent_idx, len(adapter_list.adapter_vaes) - 1)
        ]
        if not hasattr(parent_vae, "saved_distributions") or (
            parent_domain not in parent_vae.saved_distributions
        ):
            return

        # Build temporary DataLoader without shuffling, batch_size=1 (using same dataset)
        base_loader = self.train_dataloader[dataset_idx]
        dataset_obj = base_loader.dataset
        tmp_loader = DataLoader(
            dataset_obj,
            batch_size=1,
            shuffle=False,
            num_workers=4,
            worker_init_fn=_init_fn,
        )

        self.rppg_estimator.eval()
        device = self.device

        import numpy as np

        N = len(dataset_obj)
        dists = np.full((N,), np.inf, dtype=np.float32)

        layer_key = f"x_attn_{self.sample_anneal_layer}"
        pbar = tqdm(
            range(len(tmp_loader)),
            desc=f"Precomputing distances (layer {self.sample_anneal_layer})",
            leave=False,
        )
        idx_counter = 0
        for _, sample_batched in zip(pbar, tmp_loader):
            # Support dataset returning dict or tuple, current code assumes dict
            video = sample_batched["video"].to(device)
            all_inputs = {"input_clip": video}
            outputs = self.rppg_estimator(all_inputs)
            feat = outputs.get(layer_key, None)
            if feat is None:
                # Try fallback to layer 2
                feat = outputs.get("x_attn_2", None)
            if feat is None:
                # Unable to get features, keep inf
                idx_counter += 1
                continue
            try:
                dist = parent_vae.compute_distribution_distance(
                    feat, domain_name=parent_domain
                )
                dists[idx_counter] = float(dist) if dist is not None else np.inf
            except Exception as e:
                # Fallback: if failed, set to large value
                dists[idx_counter] = np.inf
            idx_counter += 1

        # Ascending sort (closer samples to parent domain come first)
        order = np.argsort(dists)
        self.sample_curriculum[dataset_idx] = {
            "dist": dists,
            "order": order,
            "N": int(N),
        }

        # Print summary
        finite = np.isfinite(dists)
        ratio_finite = float(finite.sum()) / float(N) if N > 0 else 0.0

    # ================= Adapter output distribution reference: initialization and update =================
    def _flatten_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Flatten [B, L, D] or [*, D] tensor to [N, D] (return CPU tensor)."""
        if x.dim() == 3:
            n, l, d = x.shape
            x2d = x.reshape(n * l, d)
        elif x.dim() == 2:
            x2d = x
        else:
            x2d = x.view(x.size(0), -1)
        return x2d.detach().cpu()

    def _sample_rows_cpu(self, x_cpu: torch.Tensor, num: int) -> torch.Tensor:
        """Randomly sample num rows from CPU [N,D] tensor (without replacement). If N < num, return all."""
        if x_cpu is None:
            return None
        N = x_cpu.shape[0]
        if N == 0:
            return x_cpu
        if num >= N:
            return x_cpu
        idx = torch.randperm(N)[:num]
        return x_cpu[idx]

    def _append_collect(
        self, dataset_idx: int, layer_key: str, samples_cpu: torch.Tensor
    ):
        if samples_cpu is None or samples_cpu.numel() == 0:
            return
        if dataset_idx not in self.adapter_ref_cur_collect:
            self.adapter_ref_cur_collect[dataset_idx] = {
                "x_attn_1": [],
                "x_attn_2": [],
                "x_attn_3": [],
            }
        self.adapter_ref_cur_collect[dataset_idx][layer_key].append(samples_cpu)

    @torch.no_grad()
    def _collect_adapter_outputs_from_batch(self, outputs: dict, dataset_idx: int):
        """Collect new Adapter output samples from current batch, add to current round accumulation cache (CPU)."""
        layer_map = [
            ("x_attn_1", "adapter_list_1"),
            ("x_attn_2", "adapter_list_2"),
            ("x_attn_3", "adapter_list_3"),
        ]
        for layer_key, adapter_attr in layer_map:
            if layer_key not in outputs or not hasattr(
                self.rppg_estimator, adapter_attr
            ):
                continue
            adapter_list = getattr(self.rppg_estimator, adapter_attr)
            if not hasattr(adapter_list, "adapters") or len(adapter_list.adapters) == 0:
                continue
            # Take latest adapter (new domain added adapters assumed at end)
            new_adapter = adapter_list.adapters[-1]
            x_pre = outputs[layer_key]
            new_out = new_adapter(x_pre)  # [B,L,D]
            X = self._flatten_tokens(new_out)
            Xs = self._sample_rows_cpu(X, self.adapter_ref_tokens_per_batch)
            self._append_collect(dataset_idx, layer_key, Xs)

    def _finalize_epoch_reference(self, dataset_idx: int):
        """Merge current round accumulated samples, truncate to capacity, form next round reference distribution Rt."""
        cur = self.adapter_ref_cur_collect.get(dataset_idx, None)
        if cur is None:
            return
        prev = {"x_attn_1": None, "x_attn_2": None, "x_attn_3": None}
        for layer_key in ["x_attn_1", "x_attn_2", "x_attn_3"]:
            chunks = cur.get(layer_key, [])
            if len(chunks) == 0:
                prev[layer_key] = None
                continue
            try:
                cat = torch.cat(chunks, dim=0)
            except Exception:
                # If dimensions inconsistent, skip this layer
                prev[layer_key] = None
                continue
            prev[layer_key] = self._sample_rows_cpu(cat, self.adapter_ref_capacity)
        self.adapter_ref_prev[dataset_idx] = prev
        # Clear current round accumulation
        self.adapter_ref_cur_collect[dataset_idx] = {
            "x_attn_1": [],
            "x_attn_2": [],
            "x_attn_3": [],
        }

    @torch.no_grad()
    def _build_initial_adapter_reference(self, dataset_idx: int):
        """After inheritance completion and entering new domain training, use several batches to build initial reference distribution R0."""
        if dataset_idx == 0:
            return
        loader = self.train_dataloader[dataset_idx]
        if loader is None:
            return
        was_training = self.rppg_estimator.training
        self.rppg_estimator.eval()
        # Reset current round accumulation
        self.adapter_ref_cur_collect[dataset_idx] = {
            "x_attn_1": [],
            "x_attn_2": [],
            "x_attn_3": [],
        }
        it = iter(loader)
        max_batches = max(1, self.adapter_ref_init_batches)
        for _ in range(max_batches):
            try:
                batch = next(it)
            except StopIteration:
                break
            video = batch["video"].to(self.device)
            outputs = self.rppg_estimator({"input_clip": video})
            self._collect_adapter_outputs_from_batch(outputs, dataset_idx)
        # Solidify as R0
        self._finalize_epoch_reference(dataset_idx)
        if was_training:
            self.rppg_estimator.train()

    def _coverage_at_epoch(self, dataset_idx: int, epoch: int) -> float:
        """Calculate current epoch coverage rate q_t (linearly rising from q0 to 1.0)."""
        if (
            self.sample_anneal_warmup_epochs is None
            or self.sample_anneal_warmup_epochs < 0
        ):
            T = max(1, int(self.all_epochs[dataset_idx] * 0.5))
        else:
            T = max(1, int(self.sample_anneal_warmup_epochs))
        q0 = min(max(self.sample_anneal_q0, 0.0), 1.0)
        t = max(0, epoch)
        if t >= T:
            return 1.0
        return q0 + (1.0 - q0) * (t / float(T))

    def _subset_indices_for_epoch(self, dataset_idx: int, epoch: int):
        """Generate current epoch sample subset indices based on coverage rate and exploration ratio."""
        import numpy as np

        info = self.sample_curriculum.get(dataset_idx, None)
        if info is None:
            return None
        order = info["order"]
        N = info["N"]
        if N == 0:
            return None

        # --- Scheme C: dynamic q_eff based on W (sample distances) ---
        # Linear baseline q_lin
        q_lin = self._coverage_at_epoch(dataset_idx, epoch)

        # read per-sample distances (precomputed)
        dists = info.get("dist", None)
        finite_mask = None
        if dists is None:
            # fallback to linear
            q_eff = q_lin
        else:
            dists = np.array(dists)
            finite_mask = np.isfinite(dists)
            N_finite = int(finite_mask.sum())
            finite_ratio = float(N_finite) / float(N) if N > 0 else 0.0

            # fallback if not enough finite samples
            if (
                finite_ratio < self.sample_finite_ratio_thresh
                or epoch < self.sample_warmup_epochs
            ):
                q_eff = q_lin
            else:
                # summary statistic (use mean over finite dists)
                W_t = float(dists[finite_mask].mean())

                # init per-dataset stats if missing
                stats = self.sample_stats.get(dataset_idx, None)
                if stats is None:
                    stats = {
                        "W_ema": W_t,
                        "med_ema": float(np.median(dists[finite_mask])),
                        "mad_ema": float(
                            np.median(
                                np.abs(
                                    dists[finite_mask] - np.median(dists[finite_mask])
                                )
                            )
                        ),
                    }
                else:
                    stats["W_ema"] = float(
                        self.sample_w_ema_alpha * stats.get("W_ema", W_t)
                        + (1.0 - self.sample_w_ema_alpha) * W_t
                    )
                    # update med/mad EMA
                    med_now = float(np.median(dists[finite_mask]))
                    mad_now = float(np.median(np.abs(dists[finite_mask] - med_now)))
                    stats["med_ema"] = float(
                        self.sample_w_ema_alpha * stats.get("med_ema", med_now)
                        + (1.0 - self.sample_w_ema_alpha) * med_now
                    )
                    stats["mad_ema"] = float(
                        self.sample_w_ema_alpha * stats.get("mad_ema", mad_now)
                        + (1.0 - self.sample_w_ema_alpha) * mad_now
                    )

                # store back
                self.sample_stats[dataset_idx] = stats

                W_ema = stats["W_ema"]
                med_ema = stats["med_ema"]
                mad_ema = max(stats["mad_ema"], 1e-6)

                # robust z-score
                z_t = (W_ema - med_ema) / (mad_ema + 1e-6)

                # sigmoid mapping p in [0,1], larger W -> larger p
                s = float(self.sample_w_sigmoid_s)
                z0 = float(self.sample_w_sigmoid_z0)
                p_t = 1.0 / (1.0 + np.exp(-s * (z_t - z0)))

                # mix with linear baseline
                beta = float(self.sample_w_beta)
                q_eff = q_lin + (1.0 - q_lin) * beta * float(p_t)

        # clip q_eff
        q_eff = max(min(q_eff, 1.0), float(self.sample_anneal_q0))

        # compute core size
        k_min = max(1, int(round(self.sample_k_min_ratio * N)))
        k = int(max(k_min, min(N, round(q_eff * N))))
        core_idx = order[:k]

        # Exploration: randomly take some from remaining samples
        explore_ratio = min(max(self.sample_anneal_explore, 0.0), 1.0)
        extra = int(round(explore_ratio * N))
        if extra > 0 and k < N:
            rest = order[k:]
            rng = np.random.RandomState(seed=epoch + 12345)
            if extra >= len(rest):
                explore_idx = rest
            else:
                choose = rng.choice(len(rest), size=extra, replace=False)
                explore_idx = rest[choose]
            subset = np.unique(np.concatenate([core_idx, explore_idx]))
        else:
            subset = core_idx

        return subset.tolist()

    def update_curriculum_dataloader_for_epoch(self, dataset_idx: int, epoch: int):
        """Rebuild DataLoader with SubsetRandomSampler for current epoch."""
        if not self.enable_sample_anneal or dataset_idx == 0:
            return
        if dataset_idx not in self.sample_curriculum:
            return
        subset_idx = self._subset_indices_for_epoch(dataset_idx, epoch)
        if not subset_idx:
            return

        # Visibility log: print coverage rate, subset size, etc., to confirm annealing is working
        try:
            info = self.sample_curriculum.get(dataset_idx, None)
            if info is not None:
                N = info.get("N", len(subset_idx))
                q_t = self._coverage_at_epoch(dataset_idx, epoch)
                k_est = int(max(1, min(N, round(q_t * N))))
                exp_est = int(round(self.sample_anneal_explore * N))
        except Exception:
            pass

        base_loader = self.train_dataloader[dataset_idx]
        dataset_obj = base_loader.dataset
        batch_size = self.batch_size
        num_workers = 4

        sampler = SubsetRandomSampler(subset_idx)
        new_loader = DataLoader(
            dataset_obj,
            batch_size=batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=num_workers,
            worker_init_fn=_init_fn,
        )
        self.train_dataloader[dataset_idx] = new_loader

    def expand_adapters_for_new_domain(self, dataset_idx):
        """Expand adapters for new domain (original method, keep backward compatibility)"""
        if (
            hasattr(self.rppg_estimator, "expend")
            and hasattr(self.rppg_estimator, "use_adapters")
            and self.rppg_estimator.use_adapters
        ):
            # Get adapter info before expansion
            if hasattr(self.rppg_estimator, "get_adapter_info"):
                old_adapter_info = self.rppg_estimator.get_adapter_info()
            else:
                old_adapter_info = "No adapter info available"

            # Expand adapter
            new_adapter_indices = self.rppg_estimator.expend()
            self.expanded_adapter_layers = {"x_attn_1", "x_attn_2", "x_attn_3"}
            if not hasattr(self.rppg_estimator, "parent_info"):
                self.rppg_estimator.parent_info = {}
            for layer_idx in (1, 2, 3):
                adapter_list = getattr(
                    self.rppg_estimator, f"adapter_list_{layer_idx}"
                )
                self.rppg_estimator.parent_info[f"layer_{layer_idx}"] = {
                    "parent_idx": max(0, len(adapter_list.adapters) - 2)
                }
            # === Continual learning expansion mechanism ===
            # Take adapter_list_1 as example, can extend to multiple layers similarly
            alist = self.rppg_estimator.adapter_list_1
            if len(alist.adapters) > 1:
                best_adapter = alist.adapters[-2]
                new_adapter = alist.adapters[-1]
                new_adapter.load_state_dict(best_adapter.state_dict())
                best_vae = alist.adapter_vaes[-2]
                new_vae = alist.adapter_vaes[-1]
                new_vae.load_state_dict(best_vae.state_dict())
                # Add new routing vector (column vector), consistent with the routing vector format in AdapterList
                alist.adapter_weights.append(
                    torch.nn.Parameter(
                        torch.randn(alist.dim, device=self.device) * 1e-3
                    )
                )
                # Only freeze old Adapter/VAE, keep backbone trainable
                for adapter in alist.adapters[:-1]:
                    for param in adapter.parameters():
                        param.requires_grad = False
                for vae in alist.adapter_vaes[:-1]:
                    for param in vae.parameters():
                        param.requires_grad = False
                # Train new Adapter/VAE and all routing weights
                for param in new_adapter.parameters():
                    param.requires_grad = True
                for param in new_vae.parameters():
                    param.requires_grad = True
                for w in alist.adapter_weights:
                    w.requires_grad = True

            # Get adapter info after expansion
            if hasattr(self.rppg_estimator, "get_adapter_info"):
                new_adapter_info = self.rppg_estimator.get_adapter_info()
            else:
                new_adapter_info = "No adapter info available"

            # Print adapter expansion information
            self.logger.info(
                f"Expanded adapters for domain {self.all_datasets[dataset_idx]}:"
            )
            self.logger.info(f"Old adapter info: {old_adapter_info}")
            self.logger.info(f"New adapter indices: {new_adapter_indices}")
            self.logger.info(f"New adapter info: {new_adapter_info}")

            # Rebuild optimizer to include new adapter parameters
            self.optimizer = build_optimizer(self.args, self.rppg_estimator)
            self.scheduler = build_scheduler(self.args, self.optimizer)

            # Ensure the expanded model is fully on the correct device
            self.rppg_estimator.to(self.device)

            # Update adapter configuration information
            current_domain = self.all_datasets[dataset_idx]
            current_config = (
                self.rppg_estimator.adapter_list_1.get_num_adapters(),
                self.rppg_estimator.adapter_list_2.get_num_adapters(),
                self.rppg_estimator.adapter_list_3.get_num_adapters(),
            )
            self.domain_adapter_configs[current_domain] = current_config
            self.logger.info(
                f"Updated adapter config for {current_domain}: {current_config}"
            )

            # Initialize new adapter feature storage
            try:
                self.rppg_estimator.initialize_new_adapters(
                    self.train_dataloader[dataset_idx]
                )
            except Exception as e:
                self.logger.warning(f"Failed to initialize new adapter features: {e}")
                print(f"Warning: Failed to initialize new adapter features: {e}")

            self.logger.info(f"Rebuilt optimizer and scheduler for expanded adapters")

            return True
        else:
            self.logger.info(f"Model does not support adapter expansion")
            return False

    def save_domain_distributions(self, dataset_idx, epoch):
        """Save current domain VAE distribution information"""
        if not (
            hasattr(self.rppg_estimator, "save_domain_distributions")
            and hasattr(self.rppg_estimator, "use_adapters")
            and self.rppg_estimator.use_adapters
        ):
            return

        try:
            domain_name = self.all_datasets[dataset_idx]
            self.rppg_estimator.save_domain_distributions(
                self.val_dataloader[dataset_idx], domain_name=domain_name
            )

            self.logger.info(
                f"Saved VAE distributions for domain {domain_name} at epoch {epoch}"
            )
            if self.distribution_store_path:
                self.rppg_estimator.persist_domain_fingerprints(
                    domain_name, self.distribution_store_path
                )
                self.rppg_estimator.reload_persisted_distributions(
                    self.distribution_store_path
                )

        except Exception as e:
            self.logger.error(f"Error saving VAE distributions: {e}")
            print(f"Error saving VAE distributions: {e}")

    def prepare_train(self, start_dataset, start_epoch, continue_log):
        """Prepares the training process.

        Sets up the necessary directories for saving checkpoints and logs. Initializes the logger for logging training progress.
        Copies the current file to the save path. Loads the checkpoint if starting from a specific epoch.
        Sets the rppg_estimator to train mode.

        Args:
            start_dataset (int): The starting dataset index.
            start_epoch (int): The starting epoch for training.
            continue_log (str): The name of the log to continue from.

        Raises:
            Exception: If the rppg_estimator checkpoint file for the previous epoch is not found.

        Returns:
            None
        """
        if start_dataset != 0 or start_epoch != 0:
            self.save_path = f"{self.args.save_path}/{continue_log}"
            self.run_date = continue_log

        self.save_ckpt_path = f"{self.save_path}/ckpt"
        if not os.path.exists(self.save_ckpt_path):
            os.makedirs(self.save_ckpt_path)

        self.distribution_store_path = os.path.join(
            self.save_path, "distribution_store"
        )
        if not os.path.exists(self.distribution_store_path):
            os.makedirs(self.distribution_store_path)

        # Force use logs folder under SEDA_Nor directory
        code_dir = os.path.dirname(os.path.abspath(__file__))
        logs_dir = os.path.join(code_dir, "logs")
        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir)

        log_filename = os.path.join(
            logs_dir,
            f"{self.args.model}_{self.args.dataset}_{self.args.num_rppg}_{self.run_date}.log",
        )
        logging.basicConfig(filename=log_filename, format="%(message)s", filemode="a")
        self.logger = logging.getLogger(log_filename)
        self.logger.setLevel(logging.INFO)

        # Record command line arguments at log beginning
        if (
            start_dataset == 0 and start_epoch == 0
        ):  # Only record when starting new training
            import sys

            cmd_args = " ".join(
                sys.argv[1:]
            )  # Get command line arguments except script name
            self.logger.info(f"=" * 80)
            self.logger.info(
                f"EXPERIMENT COMMAND LINE: python {sys.argv[0]} {cmd_args}"
            )
            self.logger.info(
                f'EXPERIMENT START TIME: {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}'
            )
            self.logger.info(f"=" * 80)

        ## save proj_file to save_path
        cur_file = os.getcwd()
        cur_file_name = cur_file.split("/")[-1]
        shutil.copytree(
            f"{cur_file}", f"{self.save_path}/{cur_file_name}", dirs_exist_ok=True
        )

        if start_dataset != 0 or start_epoch != 0:
            # If starting from non-first dataset, need to expand adapters to corresponding number first
            if start_dataset > 0:
                self.logger.info(
                    f"Expanding adapters for resuming from dataset {start_dataset}"
                )
                for i in range(start_dataset):
                    # Choose expansion method based on intelligent expansion parameters
                    if getattr(self.args, "use_intelligent_expand", False):
                        self.intelligent_expand_adapters_for_new_domain(i)
                    else:
                        self.expand_adapters_for_new_domain(i)
                self.is_first_domain = False

            # Load model parameters
            dataset_to_load = start_dataset - 1 if start_dataset > 0 else start_dataset
            checkpoint_path = f"{self.save_ckpt_path}/rppg_estimator_{self.all_datasets[dataset_to_load]}_best.pth"

            if not os.path.exists(checkpoint_path):
                raise Exception(
                    f"rppg_estimator ckpt file not found: {checkpoint_path}"
                )

            # Use map_location to ensure loading to correct device, add weights_only=True
            checkpoint = torch.load(
                checkpoint_path, map_location=self.device, weights_only=True
            )
            missing_keys, unexpected_keys = self.rppg_estimator.load_state_dict(
                checkpoint, strict=False
            )

            if missing_keys:
                self.logger.info(
                    f"Missing keys when loading checkpoint: {missing_keys}"
                )
            if unexpected_keys:
                self.logger.info(
                    f"Unexpected keys when loading checkpoint: {unexpected_keys}"
                )

        ## Ensure model is on correct device
        self.rppg_estimator.to(self.device)
        self.rppg_estimator.train()

    def update_best(
        self,
        cur_dataset,
        epoch,
        hr_pred,
        hr_gt,
        val_type="video",
        store_best=True,
        log_prefix=None,
    ):
        """Updates the best validation metrics and saves the model if the current metrics are better.

        Calculates the mean absolute error (MAE), root mean squared error (RMSE), standard deviation (SD),
        and Pearson correlation coefficient (R) between the predicted and ground truth heart rates.
        If the current MAE is lower than the best MAE, updates the best MAE, RMSE, epoch, SD, and R,
        and saves the model.

        Args:
            cur_dataset (int): The current dataset index.
            epoch (int): The current epoch number.
            hr_pred (List[float]): The predicted heart rates.
            hr_gt (List[float]): The ground truth heart rates.
            val_type (str, optional): The type of validation. Defaults to 'video'.
            store_best (bool, optional): Whether to update best metrics. Defaults to True.
            log_prefix (str, optional): Prefix for logging. Defaults to None.

        Returns:
            None
        """
        cur_mae = np.mean(np.abs(np.array(hr_gt) - np.array(hr_pred)))
        cur_rmse = np.sqrt(np.mean(np.square(np.array(hr_gt) - np.array(hr_pred))))
        cur_sd = np.std(np.array(hr_gt) - np.array(hr_pred))
        cur_r = pearson_correlation_coefficient(np.array(hr_gt), np.array(hr_pred))

        if store_best and cur_mae < self.best_val_mae[cur_dataset]:
            self.best_val_mae[cur_dataset] = cur_mae
            self.best_val_rmse[cur_dataset] = cur_rmse
            self.best_epoch[cur_dataset] = epoch
            self.best_sd[cur_dataset] = cur_sd
            self.best_r[cur_dataset] = cur_r
            # save the model
            torch.save(
                self.rppg_estimator.state_dict(),
                os.path.join(
                    self.save_ckpt_path,
                    f"rppg_estimator_{self.all_datasets[cur_dataset]}_best.pth",
                ),
            )

        # Always update last epoch metrics
        self.last_val_mae[cur_dataset] = cur_mae
        self.last_val_rmse[cur_dataset] = cur_rmse
        self.last_sd[cur_dataset] = cur_sd
        self.last_r[cur_dataset] = cur_r

        log_msg = (
            log_prefix
            if log_prefix
            else f"evaluate epoch {epoch}, total val {len(hr_gt)} ----------------------------------"
        )
        self.logger.info(log_msg)
        self.logger.info(
            f"{val_type}-level mae of vq: {np.mean(np.abs(np.array(hr_gt) - np.array(hr_pred)))}"
        )
        self.logger.info(
            f"{val_type}-level cur mae: {cur_mae:.2f}, cur rmse: {cur_rmse:.2f}, cur sd: {cur_sd:.2f}, cur r: {cur_r:.4f}"
        )
        if store_best:
            self.logger.info(
                f"{val_type}-level best mae of vq: {self.best_val_mae[cur_dataset]:.2f}, best rmse: {self.best_val_rmse[cur_dataset]:.2f}, best epoch: {self.best_epoch[cur_dataset]}, "
                f"best sd: {self.best_sd[cur_dataset]:.2f}, best r: {self.best_r[cur_dataset]:.4f}"
            )
        self.logger.info(
            "------------------------------------------------------------------"
        )

    def evaluate_video(
        self, cur_dataset, epoch=0, ckpt_path=None, store_best=True, log_prefix=None
    ):
        """Evaluates the video data using the current training model checkpoint.

        Load the latest checkpoint of the current training domain, then evaluate on all datasets.
        This ensures evaluation always uses the current epoch's model state.

        Args:
            cur_dataset (int): The current dataset index being evaluated.
            epoch (int, optional): The current epoch number. Defaults to 0.
            ckpt_path (str, optional): Path to checkpoint to load. Defaults to None.
            store_best (bool, optional): Whether to update best metrics. Defaults to True.
            log_prefix (str, optional): Prefix for logging. Defaults to None.

        Returns:
            None
        """
        hr_gt = []
        hr_pred = []

        # Load checkpoint if provided
        if ckpt_path is not None and os.path.exists(ckpt_path):
            try:
                self.logger.info(f"Loading checkpoint from: {ckpt_path}")

                # Load checkpoint
                checkpoint = torch.load(
                    ckpt_path, map_location=self.device, weights_only=True
                )
                missing_keys, unexpected_keys = self.rppg_estimator.load_state_dict(
                    checkpoint, strict=False
                )

                if missing_keys:
                    self.logger.warning(
                        f"Missing keys when loading checkpoint: {len(missing_keys)} keys"
                    )

                if unexpected_keys:
                    self.logger.warning(
                        f"Unexpected keys when loading checkpoint: {len(unexpected_keys)} keys"
                    )

                if not missing_keys and not unexpected_keys:
                    self.logger.info(f"Successfully loaded checkpoint for evaluation")

            except Exception as e:
                self.logger.error(f"Error loading checkpoint: {e}")
        elif ckpt_path is not None:
            self.logger.warning(
                f"Checkpoint not found: {ckpt_path}, using current model state"
            )

        with torch.no_grad():
            num_clip = 3
            for sample_batched in tqdm(self.val_dataloader[cur_dataset]):
                # get the inputs
                inputs, ecg, clip_average_HR = (
                    sample_batched["video"].to(self.device),
                    sample_batched["ecg"].to(self.device),
                    sample_batched["clip_avg_hr"].to(self.device),
                )

                input_len = inputs.shape[2]
                input_len = input_len - input_len % (num_clip * 4)
                clip_len = input_len // num_clip
                inputs = inputs[:, :, :input_len, :, :]
                ecg = ecg[:, :input_len]

                # If clip length differs from training, need to create an adapted model
                if clip_len != self.args.num_rppg:
                    # Create model instance for evaluation, set correct clip length
                    new_args = deepcopy(self.args)
                    new_args.num_rppg = clip_len

                    # Get current training model's adapter configuration
                    current_adapter_config = self.get_current_adapter_config()

                    # Build adapted model
                    val_rppg_estimator = self.build_model_with_adapter_config(
                        new_args, current_adapter_config
                    )

                    # Copy current training model parameters to evaluation model
                    val_rppg_estimator.load_state_dict(
                        self.rppg_estimator.state_dict(), strict=False
                    )
                    val_rppg_estimator.eval()

                    current_model = val_rppg_estimator
                else:
                    # Directly use current training model
                    self.rppg_estimator.eval()
                    current_model = self.rppg_estimator

                psd_gt_total = torch.tensor(0.0, device=self.device)
                psd_pred_total = torch.tensor(0.0, device=self.device)

                for idx in range(num_clip):
                    inputs_iter = inputs[
                        :, :, idx * clip_len : (idx + 1) * clip_len, :, :
                    ]
                    # ecg_iter = ecg[:, idx*clip_len : (idx+1)*clip_len]
                    ecg_iter = ecg[0, idx * clip_len : (idx + 1) * clip_len].cpu()
                    ecg_iter = np.convolve(ecg_iter, np.ones(5) / 5, mode="same")
                    ecg_iter = torch.from_numpy(ecg_iter).unsqueeze(0).to(self.device)

                    psd_gt = cal_psd_hr(ecg_iter, self.frame_rate, return_type="psd")
                    psd_gt_total += psd_gt.view(-1).max(0)[1] + 40

                    ## for rppg_estimator:
                    all_inputs = {
                        "input_clip": inputs_iter,
                    }
                    outputs = current_model(all_inputs)
                    rPPG = outputs["rPPG"]

                    psd_pred = cal_psd_hr(rPPG[0], self.frame_rate, return_type="psd")
                    psd_pred_total += psd_pred.view(-1).max(0)[1] + 40

                hr_pred.append((psd_pred_total / num_clip).cpu().item())
                hr_gt.append((psd_gt_total / num_clip).cpu().item())

        # Restore training mode
        self.rppg_estimator.train()

        ## save the results
        self.update_best(
            cur_dataset,
            epoch,
            hr_pred,
            hr_gt,
            val_type="video",
            store_best=store_best,
            log_prefix=log_prefix,
        )

    def print_vae_training_status(self):
        """Print VAE training status"""
        # Get all AdapterList in model
        adapter_lists = []
        for layer in [1, 2, 3]:
            attr_name = f"adapter_list_{layer}"
            if hasattr(self.rppg_estimator, attr_name):
                adapter_list = getattr(self.rppg_estimator, attr_name)
                if adapter_list is not None:
                    adapter_lists.append((f"layer_{layer}", adapter_list))

        if not adapter_lists:
            self.logger.info("VAE Status: No adapters found")
            return

        self.logger.info("VAE Training Status:")
        for layer_name, adapter_list in adapter_lists:
            if hasattr(adapter_list, "adapter_vaes") and hasattr(
                adapter_list, "adapter_input_features"
            ):
                num_vaes = len(adapter_list.adapter_vaes)
                if num_vaes > 0:
                    training_status = (
                        "Training" if adapter_list.adapter_vaes[0].training else "Eval"
                    )
                    self.logger.info(
                        f"  {layer_name}: {num_vaes} VAEs - Status: {training_status}"
                    )

                    # Calculate VAE loss for each adapter separately
                    for i, vae in enumerate(adapter_list.adapter_vaes):
                        adapter_features = adapter_list.get_adapter_input_features(i)

                        if adapter_features is not None:
                            try:
                                total_loss, recon_loss, kl_loss = (
                                    self.compute_single_vae_loss(adapter_features, vae)
                                )
                                self.logger.info(
                                    f"    Adapter {i}: Recon: {recon_loss.item():.4f}, KL: {kl_loss.item():.4f}, Total: {total_loss.item():.4f}"
                                )
                            except Exception as e:
                                self.logger.info(f"    Adapter {i}: Error - {e}")
                        else:
                            self.logger.info(
                                f"    Adapter {i}: No features stored (newly added or not yet used)"
                            )
                else:
                    self.logger.info(f"  {layer_name}: No VAEs initialized")
            else:
                self.logger.info(f"  {layer_name}: No VAE support")

    def compute_single_vae_loss(self, features, vae):
        """
        Compute reconstruction loss for single VAE

        Args:
            features (torch.Tensor): Input features [batch_size, seq_len, feature_dim]
            vae (nn.Module): VAE model

        Returns:
            tuple: (total_loss, recon_loss, kl_loss)
        """
        # Reshape features to VAE input format [batch_size * seq_len, feature_dim]
        batch_size, seq_len, feature_dim = features.shape
        # Use reshape() instead of view(), it automatically handles memory continuity issues
        features_flat = features.reshape(-1, feature_dim)

        # VAE forward pass - remove torch.no_grad() to keep gradients
        recon_x, mu, log_var = vae(features_flat)

        # Calculate reconstruction loss (MSE)
        recon_loss = F.mse_loss(recon_x, features_flat, reduction="mean")

        # Calculate KL divergence loss
        kl_loss = (
            -0.5
            * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            / features_flat.size(0)
        )

        # Total VAE loss
        kl_weight = getattr(
            self.args, "vae_kl_weight", 0.001
        )  # Use weight from parameters
        total_loss = recon_loss + kl_weight * kl_loss

        return total_loss, recon_loss, kl_loss

    def compute_vae_losses(self):
        """Compute losses for all VAEs"""
        total_vae_loss = 0.0
        vae_count = 0

        # Get all AdapterList in model
        adapter_lists = []
        for layer in [1, 2, 3]:
            attr_name = f"adapter_list_{layer}"
            if hasattr(self.rppg_estimator, attr_name):
                adapter_lists.append(getattr(self.rppg_estimator, attr_name))

        for adapter_list in adapter_lists:
            for i, vae in enumerate(adapter_list.adapter_vaes):
                adapter_features = adapter_list.get_adapter_input_features(i)
                if adapter_features is not None:
                    adapter_features = adapter_features.detach()
                    # Only calculate loss for VAEs that need training
                    if any(p.requires_grad for p in vae.parameters()):
                        try:
                            total_loss, recon_loss, kl_loss = (
                                self.compute_single_vae_loss(adapter_features, vae)
                            )
                            total_vae_loss += total_loss
                            vae_count += 1
                        except Exception as e:
                            self.logger.warning(f"Error computing VAE loss: {e}")
                else:
                    continue  # Skip uninitialized features
        # Return average VAE loss
        if vae_count > 0:
            return total_vae_loss / vae_count
        else:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

    def train_one_epoch(self, cur_dataset, epoch):
        """Trains the model for one epoch.

        Performs forward and backward propagation, calculates losses, and updates the model parameters.
        Updates the loss and metrics savers, and saves the ECG images periodically during training.

        Args:
            cur_dataset (int): The current dataset index.
            epoch (int): The current epoch number.

        Returns:
            dict: The training losses.
        """
        # sourcery skip: merge-dict-assign
        with tqdm(range(len(self.train_dataloader[cur_dataset]))) as pbar:
            for iter_idx, sample_batched in zip(
                pbar, self.train_dataloader[cur_dataset]
            ):
                # Ensure all input data are on correct device
                inputs = sample_batched["video"].to(self.device)
                ecg = sample_batched["ecg"].to(self.device)
                clip_average_HR = sample_batched["clip_avg_hr"].to(self.device)

                self.optimizer.zero_grad()
                # forward + backward + optimize (unified model forward call)
                all_inputs = {"input_clip": inputs}
                outputs = self.rppg_estimator(all_inputs)
                rPPG = outputs["rPPG"]
                ## calculate loss
                train_losses = {}
                train_losses["np_loss"] = self.loss_funcs["np_loss"](rPPG, ecg)
                fre_loss, kl_loss, train_mae = self.loss_funcs["ce_loss"](
                    rPPG, clip_average_HR
                )
                train_losses["ce_loss"] = fre_loss + kl_loss
                # Add VAE loss calculation
                if "vae_loss" in self.loss_funcs:
                    if hasattr(self.rppg_estimator, "get_vae_losses"):
                        vae_losses = self.rppg_estimator.get_vae_losses()
                        if vae_losses:
                            train_losses["vae_loss"] = sum(vae_losses) / len(vae_losses)
                        else:
                            train_losses["vae_loss"] = torch.tensor(
                                0.0, device=self.device, requires_grad=True
                            )
                    else:
                        train_losses["vae_loss"] = self.compute_vae_losses()
                # PGP path stability: constrain expanded adapters toward their fixed inherited parents.
                prog_loss = torch.tensor(0.0, device=self.device)
                expanded_layers = getattr(self, "expanded_adapter_layers", set())
                if cur_dataset > 0 and expanded_layers:
                    # The paper applies constraint annealing over the first half
                    # of training, then keeps the full-data objective thereafter.
                    T_total = max(1, self.all_epochs[cur_dataset])
                    curriculum_progress = min(
                        max(float(epoch), 0.0) / max(1.0, 0.5 * T_total), 1.0
                    )
                    kappa = float(self.progressive_beta_gamma)
                    lambda_t = float(np.exp(-kappa * curriculum_progress))

                    def sliced_wasserstein_distance(X, Y, K=16, device=None):
                        # X, Y: [N, D] torch.Tensor
                        # K: number of directions
                        N = X.shape[0]
                        D = X.shape[1]
                        device = device or X.device
                        swd = torch.tensor(0.0, device=device)
                        for _ in range(K):
                            v = torch.randn(D, device=device)
                            v = v / (v.norm() + 1e-8)
                            proj_X = (X @ v).sort()[0]
                            proj_Y = (Y @ v).sort()[0]
                            # If sample counts differ, pad to match
                            if proj_X.shape[0] > proj_Y.shape[0]:
                                proj_Y = F.pad(
                                    proj_Y,
                                    (0, proj_X.shape[0] - proj_Y.shape[0]),
                                    mode="constant",
                                    value=0,
                                )
                            elif proj_X.shape[0] < proj_Y.shape[0]:
                                proj_X = F.pad(
                                    proj_X,
                                    (0, proj_Y.shape[0] - proj_X.shape[0]),
                                    mode="constant",
                                    value=0,
                                )
                            swd += torch.mean(torch.abs(proj_X - proj_Y))
                        return swd / K

                    for layer_key, adapter_attr in zip(
                        ["x_attn_1", "x_attn_2", "x_attn_3"],
                        ["adapter_list_1", "adapter_list_2", "adapter_list_3"],
                    ):
                        if layer_key not in expanded_layers:
                            continue
                        if layer_key in outputs and hasattr(
                            self.rppg_estimator, adapter_attr
                        ):
                            adapter_list = getattr(self.rppg_estimator, adapter_attr)
                            if (
                                not hasattr(adapter_list, "adapters")
                                or len(adapter_list.adapters) == 0
                            ):
                                continue
                            # Compare the new adapter against its fixed inherited parent.
                            new_adapter = adapter_list.adapters[-1]
                            layer_idx = int(layer_key.rsplit("_", 1)[1])
                            parent_info = getattr(
                                self.rppg_estimator, "parent_info", {}
                            ).get(f"layer_{layer_idx}", {})
                            parent_idx = parent_info.get("parent_idx")
                            if parent_idx is None or not (
                                0 <= int(parent_idx) < len(adapter_list.adapters) - 1
                            ):
                                continue
                            parent_adapter = adapter_list.adapters[int(parent_idx)]
                            x_pre = outputs[layer_key]
                            new_out = new_adapter(x_pre).reshape(
                                -1, x_pre.shape[-1]
                            )
                            with torch.no_grad():
                                parent_out = parent_adapter(x_pre).reshape(
                                    -1, x_pre.shape[-1]
                                )
                            if self.progressive_reg_mode == "swd":
                                prog_loss = prog_loss + sliced_wasserstein_distance(
                                    new_out,
                                    parent_out,
                                    K=self.progressive_swd_K,
                                    device=self.device,
                                )
                            else:
                                prog_loss = prog_loss + F.mse_loss(
                                    new_out.mean(dim=0, keepdim=True),
                                    parent_out.mean(dim=0, keepdim=True),
                                )
                    train_losses["progressive_reg"] = (
                        lambda_t * self.progressive_reg_weight * prog_loss
                    )
                else:
                    train_losses["progressive_reg"] = torch.tensor(
                        0.0, device=self.device
                    )

                # Accumulate current batch "student output" as current round samples, used to build next round reference distribution
                try:
                    with torch.no_grad():
                        self._collect_adapter_outputs_from_batch(outputs, cur_dataset)
                except Exception:
                    pass
                total_loss = sum(
                    train_losses[key] * self.loss_funcs_weight.get(key, 1.0)
                    for key in train_losses
                )
                total_loss.backward()
                self.optimizer.step()

                ## update loss saver and metrics saver
                train_metrics = {
                    "mae": train_mae,
                }
                update_avg_meters(self.loss_meters, train_losses, self.batch_size)
                update_avg_meters(self.metrics_meters, train_metrics, self.batch_size)

                mini_batch_info = f'[{self.all_datasets[cur_dataset]}]epoch : {epoch:0>3}, mini-batch : {iter_idx:0>4}, lr = {self.optimizer.param_groups[0]["lr"]:.5f}'
                loss_info = ", ".join(
                    [
                        f"{key} = {self.loss_meters[key].avg:.4f}"
                        for key in self.loss_meters.keys()
                    ]
                )
                metrics_info = ", ".join(
                    [
                        f"{key} = {self.metrics_meters[key].avg:.4f}"
                        for key in self.metrics_meters.keys()
                    ]
                )

                if (
                    iter_idx % self.args.echo_batches == self.args.echo_batches - 1
                ):  # info every mini-batches
                    self.logger.info(
                        ", ".join([mini_batch_info, loss_info, metrics_info])
                    )

                    # Regularly print VAE training status (every 500 batches)
                    if "vae_loss" in self.loss_funcs and iter_idx % 10 == 1:
                        self.logger.info(
                            f"VAE Training Status at Epoch {epoch}, Batch {iter_idx}"
                        )
                        self.print_vae_training_status()

                pbar.set_description(
                    ", ".join([mini_batch_info, loss_info, metrics_info])
                )
        self.scheduler.step()
        return train_losses

    def train_one_dataset(self, cur_dataset, start_epoch):
        """Trains the model.

        Prepares the training process by setting up necessary directories and loading checkpoints.
        Performs training for each epoch, saving the model and evaluating it periodically.

        Args:
            cur_dataset (int): The current dataset index.
            start_epoch (int, optional): The starting epoch number. Defaults to 0.

        Returns:
            None
        """

        for epoch in range(start_epoch, self.all_epochs[cur_dataset]):
            self.logger.info(
                f'train dataset: {self.all_datasets[cur_dataset]} epoch: {epoch} lr: {self.optimizer.param_groups[0]["lr"]:.5f}'
            )

            # Print training parameters summary every few epochs
            if epoch % 5 == 0:  # Print every 5 epochs
                self.print_training_parameters_summary()

            # Sample annealing: dynamically rebuild sample subset by epoch
            try:
                self.update_curriculum_dataloader_for_epoch(cur_dataset, epoch)
            except Exception as e:
                self.logger.warning(f"Sample anneal update failed: {e}")

            self.train_one_epoch(cur_dataset, epoch)

            # End of round: solidify current round accumulation, form next round reference distribution Rt
            try:
                if cur_dataset > 0:
                    self._finalize_epoch_reference(cur_dataset)
            except Exception as e:
                self.logger.warning(f"Finalize adapter reference failed: {e}")

            # save the model
            torch.save(
                self.rppg_estimator.state_dict(),
                os.path.join(
                    self.save_ckpt_path,
                    f"rppg_estimator_{self.all_datasets[cur_dataset]}_{epoch}.pth",
                ),
            )

            # delete the model
            if epoch > 0 and self.args.save_mode == "best":
                os.remove(
                    os.path.join(
                        self.save_ckpt_path,
                        f"rppg_estimator_{self.all_datasets[cur_dataset]}_{epoch-1}.pth",
                    )
                )

            # First domain VAE training phase: periodically save distribution information
            if self.is_first_domain and (epoch + 1) % self.vae_training_epochs == 0:
                self.save_domain_distributions(cur_dataset, epoch)

            # Evaluate on current dataset after each epoch
            self.evaluate_video(cur_dataset, epoch)

            # Force save first epoch as best model
            if epoch == start_epoch:
                self.best_val_mae[cur_dataset] = self.last_val_mae[cur_dataset]
                self.best_val_rmse[cur_dataset] = self.last_val_rmse[cur_dataset]
                self.best_sd[cur_dataset] = self.last_sd[cur_dataset]
                self.best_r[cur_dataset] = self.last_r[cur_dataset]
                self.best_epoch[cur_dataset] = epoch
                torch.save(
                    self.rppg_estimator.state_dict(),
                    os.path.join(
                        self.save_ckpt_path,
                        f"rppg_estimator_{self.all_datasets[cur_dataset]}_best.pth",
                    ),
                )
                self.logger.info(
                    f"Forced save first epoch {epoch} as best for dataset {self.all_datasets[cur_dataset]}"
                )

    def train_continual_learning(self, start_dataset=0, start_epoch=0, continue_log=""):
        self.prepare_train(
            start_dataset=start_dataset,
            start_epoch=start_epoch,
            continue_log=continue_log,
        )
        self.logger.info(
            f"prepare train, load ckpt and block gradient, start_dataset: {self.all_datasets[start_dataset]} start_epoch: {start_epoch}, gpu: {self.gpu_list}.\n"
            f"dataset: {self.args.dataset}, num_rppg: {self.args.num_rppg}, model: {self.args.model}, loss: {self.loss_funcs_weight}.\n"
            f"batch_size: {self.batch_size}, lr: {self.args.lr}, optim: {self.args.optim}, scheduler: {self.args.scheduler}.\n"
            f"VAE Wasserstein params: normalize={self.normalize_wasserstein}, method={self.wasserstein_norm_method}."
        )

        # Output initial adapter info before starting training
        if hasattr(self.rppg_estimator, "get_adapter_info"):
            initial_adapter_info = self.rppg_estimator.get_adapter_info()
            self.logger.info(f"Initial adapter info: {initial_adapter_info}")

        for dataset in range(start_dataset, len(self.all_datasets)):
            self.logger.info(
                f"================= CL at Dataset : {self.all_datasets[dataset]} ===================="
            )

            # First domain: train VAE
            if dataset == 0 and self.is_first_domain:
                self.logger.info(
                    f"First domain training: Training VAE and adapters for {self.all_datasets[dataset]}"
                )

            # New domain: intelligent adapter expansion
            elif dataset > 0:
                # First save previous domain's distribution information
                if dataset > 0:
                    self.save_domain_distributions(
                        dataset - 1, self.best_epoch[dataset - 1]
                    )

                # Choose to load best or latest model from previous dataset as starting point for new dataset based on parameters
                if self.args.new_domain_load_mode == "best":
                    prev_ckpt = os.path.join(
                        self.save_ckpt_path,
                        f"rppg_estimator_{self.all_datasets[dataset-1]}_best.pth",
                    )
                    ckpt_type = "best"
                elif self.args.new_domain_load_mode == "latest":
                    prev_ckpt = os.path.join(
                        self.save_ckpt_path,
                        f"rppg_estimator_{self.all_datasets[dataset-1]}_{self.all_epochs[dataset-1]-1}.pth",
                    )
                    ckpt_type = "latest"
                else:
                    raise ValueError(
                        f"Invalid new_domain_load_mode: {self.args.new_domain_load_mode}. Must be 'best' or 'latest'"
                    )

                if os.path.exists(prev_ckpt):
                    try:
                        checkpoint = torch.load(
                            prev_ckpt, map_location=self.device, weights_only=True
                        )
                        self.rppg_estimator.load_state_dict(checkpoint, strict=False)
                        self.logger.info(
                            f"Loaded {ckpt_type} model from previous domain: {prev_ckpt}"
                        )
                        print(
                            f"Loaded {ckpt_type} model from previous domain: {prev_ckpt}"
                        )

                        # Reload previous domain's distribution information into model (because distribution info is not saved in state_dict)
                        self.rppg_estimator.save_domain_distributions(
                            self.val_dataloader[dataset - 1],
                            domain_name=self.all_datasets[dataset - 1],
                        )
                        self.logger.info(
                            f"Reloaded VAE distributions for previous domain: {self.all_datasets[dataset-1]}"
                        )
                        print(
                            f"Reloaded VAE distributions for previous domain: {self.all_datasets[dataset-1]}"
                        )

                    except Exception as e:
                        self.logger.error(f"Error loading {ckpt_type} checkpoint: {e}")
                        print(f"Error loading {ckpt_type} checkpoint: {e}")
                else:
                    self.logger.warning(
                        f"{ckpt_type.capitalize()} checkpoint not found: {prev_ckpt}, using current model"
                    )
                    print(
                        f"{ckpt_type.capitalize()} checkpoint not found: {prev_ckpt}, using current model"
                    )

                # Choose expansion method based on parameters
                if getattr(
                    self.args, "use_intelligent_expand", True
                ):  # Default to use intelligent expansion
                    # Reload all historical domain fingerprints into model
                    if self.distribution_store_path:
                        self.rppg_estimator.reload_persisted_distributions(
                            self.distribution_store_path
                        )
                    self.intelligent_expand_adapters_for_new_domain(dataset)
                else:
                    self.expand_adapters_for_new_domain(dataset)

                self.is_first_domain = False

            # Ensure model is on correct device
            self.rppg_estimator.to(self.device)

            # Set training mode
            self.setup_training_mode_for_domain(dataset)

            # Sample annealing: precompute parent domain distances for new domain (offline once)
            try:
                if self.enable_sample_anneal and dataset > 0:
                    self.precompute_sample_distances(dataset)
            except Exception as e:
                self.logger.warning(f"Precompute sample distances failed: {e}")

            # Print model training status
            # self.print_model_training_status()

            # Count parameter numbers
            self.count_trainable_parameters()

            # Before starting new domain, use several batches to build initial reference distribution R0 (only for second domain and later)
            try:
                if dataset > 0:
                    self._build_initial_adapter_reference(dataset)
            except Exception as e:
                self.logger.warning(f"Build initial adapter reference failed: {e}")

            for meter in self.loss_meters.values():
                meter.reset()
            for meter in self.metrics_meters.values():
                meter.reset()
            if start_dataset != 0 and dataset == start_dataset:
                self.train_one_dataset(dataset, start_epoch)
            else:
                self.train_one_dataset(dataset, 0)
            self.logger.info(
                f"================= END CL at Dataset : {self.all_datasets[dataset]} ===================="
            )
            self.logger.info(
                f"====== POST-DATASET EVAL {self.all_datasets[dataset]} ======"
            )
            # Use the best model on this dataset to evaluate on all previous datasets
            best_epoch = self.best_epoch[dataset]
            best_ckpt_path = os.path.join(
                self.save_ckpt_path,
                f"rppg_estimator_{self.all_datasets[dataset]}_best.pth",
            )
            self.evaluate_all_previous_domains(
                dataset, best_epoch, ckpt_path=best_ckpt_path
            )

            # Save distribution information after training
            self.save_domain_distributions(dataset, best_epoch)

            # Calculate and store current domain's Fisher information for next domain's adaptive freezing
            if hasattr(self.rppg_estimator, "compute_and_store_fisher"):
                self.rppg_estimator.compute_and_store_fisher(
                    self.train_dataloader[dataset],
                    domain_name=self.all_datasets[dataset],
                )

        # Unified evaluation after training all domains
        final_dataset_idx = len(self.all_datasets) - 1
        if final_dataset_idx >= 0:
            final_epoch_idx = max(0, self.all_epochs[final_dataset_idx] - 1)
            self.logger.info("====== FINAL EVAL ALL DOMAINS ======")
            self.evaluate_all_previous_domains(final_dataset_idx, final_epoch_idx)

        # Perform final continual learning analysis
        final_ckpt_path = os.path.join(self.save_ckpt_path, "rppg_estimator_final.pth")
        torch.save(self.rppg_estimator.state_dict(), final_ckpt_path)
        self.logger.info("====== FINAL MODEL EVALUATION ======")
        for dataset_idx, dataset_name in enumerate(self.all_datasets):
            final_log_prefix = f"final evaluation at dataset {dataset_name} ----------------------------------"
            self.evaluate_video(
                dataset_idx,
                epoch="final",
                ckpt_path=final_ckpt_path,
                store_best=False,
                log_prefix=final_log_prefix,
            )

        self.logger.info("========== BEST EVALUATION ANALYSIS ==========")
        self.logger.info("Best performance on each domain:")
        mae_arr = np.array(self.best_val_mae)
        rmse_arr = np.array(self.best_val_rmse)
        sd_arr = np.array(self.best_sd)
        r_arr = np.array(self.best_r)
        for idx, dataset_name in enumerate(self.all_datasets):
            self.logger.info(
                f"{dataset_name}: MAE={mae_arr[idx]:.2f}, RMSE={rmse_arr[idx]:.2f}, SD={sd_arr[idx]:.2f}, R={r_arr[idx]:.4f} (best epoch: {self.best_epoch[idx]})"
            )
        self.logger.info(
            f"Overall Average MAE: {mae_arr.mean():.2f} (std: {mae_arr.std():.2f})"
        )
        self.logger.info(
            f"Overall Average RMSE: {rmse_arr.mean():.2f} (std: {rmse_arr.std():.2f})"
        )
        self.logger.info(
            f"Overall Average SD: {sd_arr.mean():.2f} (std: {sd_arr.std():.2f})"
        )
        self.logger.info(
            f"Overall Average R: {r_arr.mean():.4f} (std: {r_arr.std():.4f})"
        )
        self.logger.info(
            "=========================================================================================="
        )

        # Calculate and log the last epoch evaluation analysis
        self.logger.info("========== LAST EVALUATION ANALYSIS ==========")
        for dataset_idx, dataset_name in enumerate(self.all_datasets):
            self.logger.info(
                f"{dataset_name}: MAE={self.last_val_mae[dataset_idx]:.2f}, RMSE={self.last_val_rmse[dataset_idx]:.2f}, SD={self.last_sd[dataset_idx]:.2f}, R={self.last_r[dataset_idx]:.4f}"
            )
        self.logger.info("Last performance on each domain:")
        self.logger.info(
            f"Overall Average MAE: {np.mean(self.last_val_mae):.2f} (std: {np.std(self.last_val_mae):.2f})"
        )
        self.logger.info(
            f"Overall Average RMSE: {np.mean(self.last_val_rmse):.2f} (std: {np.std(self.last_val_rmse):.2f})"
        )
        self.logger.info(
            f"Overall Average SD: {np.mean(self.last_sd):.2f} (std: {np.std(self.last_sd):.2f})"
        )
        self.logger.info(
            f"Overall Average R: {np.mean(self.last_r):.4f} (std: {np.std(self.last_r):.4f})"
        )
        self.logger.info(
            "=========================================================================================="
        )

        self.logger.info(f"Finished CL at Datasets: {self.all_datasets}")
        self.logger.info(
            f"video-level best mean mae: {np.mean(self.best_val_mae):.2f}, mean rmse: {np.mean(self.best_val_rmse):.2f} "
            f"mean sd: {np.mean(self.best_sd):.2f}, mean r: {np.mean(self.best_r):.4f}"
        )
        self.logger.info(
            f"video-level last mean mae: {np.mean(self.last_val_mae):.2f}, mean rmse: {np.mean(self.last_val_rmse):.2f} "
            f"mean sd: {np.mean(self.last_sd):.2f}, mean r: {np.mean(self.last_r):.4f}"
        )
        self.logger.info(
            "------------------------------------------------------------------"
        )

        # Output final adapter info after training ends
        if hasattr(self.rppg_estimator, "get_adapter_info"):
            final_adapter_info = self.rppg_estimator.get_adapter_info()
            self.logger.info(f"Final adapter info: {final_adapter_info}")

    def check_adapter_features_status(self):
        """Check adapter feature storage status"""
        if not (
            hasattr(self.rppg_estimator, "use_adapters")
            and self.rppg_estimator.use_adapters
        ):
            return

        self.logger.info("ADAPTER FEATURES STATUS")

        adapter_lists = []
        if (
            hasattr(self.rppg_estimator, "adapter_list_1")
            and self.rppg_estimator.adapter_list_1 is not None
        ):
            adapter_lists.append(("layer_1", self.rppg_estimator.adapter_list_1))
        if (
            hasattr(self.rppg_estimator, "adapter_list_2")
            and self.rppg_estimator.adapter_list_2 is not None
        ):
            adapter_lists.append(("layer_2", self.rppg_estimator.adapter_list_2))
        if (
            hasattr(self.rppg_estimator, "adapter_list_3")
            and self.rppg_estimator.adapter_list_3 is not None
        ):
            adapter_lists.append(("layer_3", self.rppg_estimator.adapter_list_3))

        for layer_name, adapter_list in adapter_lists:
            if hasattr(adapter_list, "adapter_input_features"):
                num_adapters = len(adapter_list.adapters)
                num_features = len(adapter_list.adapter_input_features)
                features_status = []

                for i, features in enumerate(adapter_list.adapter_input_features):
                    if features is not None:
                        features_status.append(f"{i}:✓({features.shape})")
                    else:
                        features_status.append(f"{i}:✗")

                self.logger.info(
                    f"  {layer_name}: {num_adapters} adapters, {num_features} feature slots"
                )
                self.logger.info(f"    Features: {', '.join(features_status)}")
            else:
                self.logger.info(f"  {layer_name}: No feature storage")

    def setup_training_mode_for_domain(self, dataset_idx):
        """Set training mode for specified domain"""
        if hasattr(self.rppg_estimator, "unfreeze_backbone"):
            self.rppg_estimator.unfreeze_backbone()

        # Adaptive freeze part of transformer backbone parameters (based on accumulated Fisher)
        if hasattr(self.rppg_estimator, "adaptive_freeze_backbone_from_stored"):
            self.rppg_estimator.adaptive_freeze_backbone_from_stored(
                getattr(self.args, "backbone_freeze_ratio", 0.0)
            )
        elif hasattr(self.rppg_estimator, "random_freeze_backbone"):
            # Fallback to random freeze
            self.rppg_estimator.random_freeze_backbone(
                getattr(self.args, "backbone_freeze_ratio", 0.0)
            )

        if dataset_idx == 0 and self.is_first_domain:
            print(f"\n========== TRAINING MODE: FIRST DOMAIN =========")
            print(f"Training: Backbone + Adapters + VAEs")
            print(f"=======================================")
            self.logger.info("First domain: Training backbone + adapters + VAEs")
        else:
            print(f"\n========== TRAINING MODE: NEW DOMAIN =========")
            print(f"Training: Backbone + New Adapters + VAEs (old adapters frozen)")
            print(f"=============================================")
            self.logger.info(
                "New domain: Training backbone + newly added adapters + VAEs"
            )

    def print_model_training_status(self):
        """Print training status of each model part"""
        self.logger.info("MODEL TRAINING STATUS")

        # Check backbone model status
        backbone_trainable = 0
        backbone_total = 0

        modules_to_check = [
            ("feature_embedding", self.rppg_estimator.feature_embedding),
            ("patch_embedding", self.rppg_estimator.patch_embedding),
            ("vit_block_head", self.rppg_estimator.vit_block_head),
            ("vit_block_last_1", self.rppg_estimator.vit_block_last_1),
            ("vit_block_last_2", self.rppg_estimator.vit_block_last_2),
            ("vit_block_last_3", self.rppg_estimator.vit_block_last_3),
            ("predictor", self.rppg_estimator.predictor),
        ]

        for name, module in modules_to_check:
            if module is not None:
                trainable = sum(
                    p.numel() for p in module.parameters() if p.requires_grad
                )
                total = sum(p.numel() for p in module.parameters())
                backbone_trainable += trainable
                backbone_total += total
                status = "✓Trainable" if trainable > 0 else "✗Frozen"
                self.logger.info(f"  {name}: {status} ({trainable}/{total} params)")

        self.logger.info(
            f"Backbone Total: {backbone_trainable}/{backbone_total} params trainable"
        )

        # Check adapter status
        if (
            hasattr(self.rppg_estimator, "use_adapters")
            and self.rppg_estimator.use_adapters
        ):
            adapter_lists = []
            if hasattr(self.rppg_estimator, "adapter_list_1"):
                adapter_lists.append(("layer_1", self.rppg_estimator.adapter_list_1))
            if hasattr(self.rppg_estimator, "adapter_list_2"):
                adapter_lists.append(("layer_2", self.rppg_estimator.adapter_list_2))
            if hasattr(self.rppg_estimator, "adapter_list_3"):
                adapter_lists.append(("layer_3", self.rppg_estimator.adapter_list_3))

            for layer_name, adapter_list in adapter_lists:
                num_adapters = len(adapter_list.adapters)
                trainable_adapters = 0
                trainable_vaes = 0

                for i, (adapter, vae) in enumerate(
                    zip(adapter_list.adapters, adapter_list.adapter_vaes)
                ):
                    adapter_trainable = (
                        sum(p.numel() for p in adapter.parameters() if p.requires_grad)
                        > 0
                    )
                    vae_trainable = (
                        sum(p.numel() for p in vae.parameters() if p.requires_grad) > 0
                    )

                    if adapter_trainable:
                        trainable_adapters += 1
                    if vae_trainable:
                        trainable_vaes += 1

                self.logger.info(
                    f"  {layer_name}: {num_adapters} adapters ({trainable_adapters} trainable), {len(adapter_list.adapter_vaes)} VAEs ({trainable_vaes} trainable)"
                )

        self.logger.info("END MODEL TRAINING STATUS")

    def count_trainable_parameters(self):
        """Count the number of trainable parameters"""
        total_params = 0
        trainable_params = 0

        for name, param in self.rppg_estimator.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()

        self.logger.info(
            f"Model parameters - Total: {total_params:,}, Trainable: {trainable_params:,}, Ratio: {trainable_params/total_params*100:.2f}%"
        )

    def print_training_parameters_summary(self):
        """Print brief summary of current training parameters"""
        backbone_trainable = 0
        adapter_trainable = 0
        vae_trainable = 0
        total_trainable = 0

        # Count backbone network
        modules_to_check = [
            self.rppg_estimator.feature_embedding,
            self.rppg_estimator.patch_embedding,
            self.rppg_estimator.vit_block_head,
            self.rppg_estimator.vit_block_last_1,
            self.rppg_estimator.vit_block_last_2,
            self.rppg_estimator.vit_block_last_3,
            self.rppg_estimator.predictor,
        ]

        for module in modules_to_check:
            if module is not None:
                backbone_trainable += sum(
                    p.numel() for p in module.parameters() if p.requires_grad
                )

        # Count adapters and VAEs
        if (
            hasattr(self.rppg_estimator, "use_adapters")
            and self.rppg_estimator.use_adapters
        ):
            adapter_lists = [
                self.rppg_estimator.adapter_list_1,
                self.rppg_estimator.adapter_list_2,
                self.rppg_estimator.adapter_list_3,
            ]

            for adapter_list in adapter_lists:
                for adapter in adapter_list.adapters:
                    adapter_trainable += sum(
                        p.numel() for p in adapter.parameters() if p.requires_grad
                    )
                for weight in adapter_list.adapter_weights:
                    if weight.requires_grad:
                        adapter_trainable += weight.numel()
                for vae in adapter_list.adapter_vaes:
                    vae_trainable += sum(
                        p.numel() for p in vae.parameters() if p.requires_grad
                    )

        total_trainable = backbone_trainable + adapter_trainable + vae_trainable

        print(f"\n========== TRAINING PARAMETERS SUMMARY ==========")
        print(
            f"Backbone: {backbone_trainable:,} params ({'✓' if backbone_trainable > 0 else '✗'})"
        )
        print(
            f"Adapters: {adapter_trainable:,} params ({'✓' if adapter_trainable > 0 else '✗'})"
        )
        print(f"VAEs: {vae_trainable:,} params ({'✓' if vae_trainable > 0 else '✗'})")
        print(f"Total Training: {total_trainable:,} params")
        print(f"===============================================")

    def evaluate_all_previous_domains(
        self, cur_dataset, epoch, store_best=True, ckpt_path=None
    ):
        """Evaluate on all previously trained domains (including current domain)

        Args:
            cur_dataset (int): Current dataset index
            epoch (int): Current epoch
            store_best (bool, optional): Whether to update best metrics. Defaults to True.
            ckpt_path (str, optional): Checkpoint path. If provided, load it, otherwise use current model. Defaults to None.
        """
        print(f"\n========== EVALUATING ALL PREVIOUS DOMAINS ==========")
        print(
            f"Current training domain: {self.all_datasets[cur_dataset]}, Epoch: {epoch}"
        )
        print(f"Evaluating on domains: {self.all_datasets[:cur_dataset+1]}")
        if ckpt_path:
            print(f"Using checkpoint: {ckpt_path}")
        else:
            print(f"Using current model state")
        self.logger.info(f"Evaluating all previous domains at epoch {epoch}")
        self.logger.info(f"Current training domain: {self.all_datasets[cur_dataset]}")
        self.logger.info(f"Evaluating on domains: {self.all_datasets[:cur_dataset+1]}")
        if ckpt_path:
            self.logger.info(f"Using checkpoint: {ckpt_path}")
        else:
            self.logger.info(f"Using current model state")

        # If checkpoint path is provided, try to load
        if ckpt_path and os.path.exists(ckpt_path):
            try:
                self.logger.info(f"Loading checkpoint from: {ckpt_path}")

                checkpoint = torch.load(
                    ckpt_path, map_location=self.device, weights_only=True
                )
                missing_keys, unexpected_keys = self.rppg_estimator.load_state_dict(
                    checkpoint, strict=False
                )

                if missing_keys:
                    self.logger.warning(
                        f"Missing keys when loading checkpoint: {len(missing_keys)} keys"
                    )

                if unexpected_keys:
                    self.logger.warning(
                        f"Unexpected keys when loading checkpoint: {len(unexpected_keys)} keys"
                    )

                if not missing_keys and not unexpected_keys:
                    self.logger.info(f"Successfully loaded checkpoint for evaluation")

            except Exception as e:
                self.logger.error(f"Error loading checkpoint: {e}")
        elif ckpt_path:
            self.logger.warning(
                f"Checkpoint not found: {ckpt_path}, using current model state"
            )
        else:
            self.logger.info(f"Using current model state for evaluation")

        # Evaluate current domain and all previous domains
        for eval_dataset_idx in range(cur_dataset + 1):
            domain_name = self.all_datasets[eval_dataset_idx]
            print(f"\n--- Evaluating on domain: {domain_name} ---")
            self.logger.info(f"Evaluating on domain: {domain_name}")

            # Use current model (checkpoint loaded or current state) to evaluate on specified domain
            self.evaluate_video(eval_dataset_idx, epoch, store_best=store_best)

            # If not the current training domain, record forgetting situation
            if eval_dataset_idx < cur_dataset:
                self.log_forgetting_analysis(eval_dataset_idx, cur_dataset, epoch)

        # Calculate and record average performance
        self.log_average_performance(cur_dataset)

        print(f"\n========== EVALUATION COMPLETED ==========")
        self.logger.info(f"All domain evaluation completed at epoch {epoch}")

    def log_forgetting_analysis(self, eval_dataset_idx, cur_dataset, epoch):
        """Record forgetting analysis

        Args:
            eval_dataset_idx (int): Index of the domain being evaluated
            cur_dataset (int): Index of the currently training domain
            epoch (int): Current epoch
        """
        eval_domain_name = self.all_datasets[eval_dataset_idx]
        cur_domain_name = self.all_datasets[cur_dataset]

        current_mae = self.best_val_mae[eval_dataset_idx]

        self.logger.info(
            f"Forgetting analysis - Domain: {eval_domain_name}, "
            f"Current MAE: {current_mae:.2f}, "
            f"Best epoch: {self.best_epoch[eval_dataset_idx]} "
            f"(evaluated while training on {cur_domain_name})"
        )

    def log_average_performance(self, cur_dataset):
        """Record average performance

        Args:
            cur_dataset (int): Current dataset index
        """
        # Calculate average performance across all domains so far
        domains_so_far = cur_dataset + 1
        avg_mae = np.mean(self.best_val_mae[:domains_so_far])
        avg_rmse = np.mean(self.best_val_rmse[:domains_so_far])
        avg_sd = np.mean(self.best_sd[:domains_so_far])
        avg_r = np.mean(self.best_r[:domains_so_far])

        print(f"\n--- Average Performance across {domains_so_far} domains ---")
        print(f"Average MAE: {avg_mae:.2f}")
        print(f"Average RMSE: {avg_rmse:.2f}")
        print(f"Average SD: {avg_sd:.2f}")
        print(f"Average R: {avg_r:.4f}")

        self.logger.info(
            f"Average performance across {domains_so_far} domains: "
            f"MAE: {avg_mae:.2f}, RMSE: {avg_rmse:.2f}, "
            f"SD: {avg_sd:.2f}, R: {avg_r:.4f}"
        )

    def final_continual_learning_analysis(self):
        """Perform final continual learning performance analysis"""
        print(f"\n========== FINAL CONTINUAL LEARNING ANALYSIS ==========")
        self.logger.info("========== FINAL CONTINUAL LEARNING ANALYSIS ==========")

        # Print best performance on each domain
        print("\nBest performance on each domain:")
        self.logger.info("Best performance on each domain:")

        for i, domain_name in enumerate(self.all_datasets):
            print(
                f"  {domain_name}: MAE={self.best_val_mae[i]:.2f}, "
                f"RMSE={self.best_val_rmse[i]:.2f}, "
                f"SD={self.best_sd[i]:.2f}, "
                f"R={self.best_r[i]:.4f} "
                f"(best epoch: {self.best_epoch[i]})"
            )

            self.logger.info(
                f"{domain_name}: MAE={self.best_val_mae[i]:.2f}, "
                f"RMSE={self.best_val_rmse[i]:.2f}, "
                f"SD={self.best_sd[i]:.2f}, "
                f"R={self.best_r[i]:.4f} "
                f"(best epoch: {self.best_epoch[i]})"
            )

        # Calculate overall statistics
        final_avg_mae = np.mean(self.best_val_mae)
        final_avg_rmse = np.mean(self.best_val_rmse)
        final_avg_sd = np.mean(self.best_sd)
        final_avg_r = np.mean(self.best_r)

        self.logger.info(
            f"Overall Average MAE: {final_avg_mae:.2f} (std: {np.std(self.best_val_mae):.2f})"
        )
        self.logger.info(
            f"Overall Average RMSE: {final_avg_rmse:.2f} (std: {np.std(self.best_val_rmse):.2f})"
        )
        self.logger.info(
            f"Overall Average SD: {final_avg_sd:.2f} (std: {np.std(self.best_sd):.2f})"
        )
        self.logger.info(
            f"Overall Average R: {final_avg_r:.4f} (std: {np.std(self.best_r):.4f})"
        )

        # Find best and worst performing domains
        best_domain_idx = np.argmin(self.best_val_mae)
        worst_domain_idx = np.argmax(self.best_val_mae)

        print(
            f"\nBest performing domain: {self.all_datasets[best_domain_idx]} (MAE: {self.best_val_mae[best_domain_idx]:.2f})"
        )
        print(
            f"Worst performing domain: {self.all_datasets[worst_domain_idx]} (MAE: {self.best_val_mae[worst_domain_idx]:.2f})"
        )

        self.logger.info(
            f"Best performing domain: {self.all_datasets[best_domain_idx]} (MAE: {self.best_val_mae[best_domain_idx]:.2f})"
        )
        self.logger.info(
            f"Worst performing domain: {self.all_datasets[worst_domain_idx]} (MAE: {self.best_val_mae[worst_domain_idx]:.2f})"
        )

        print(f"=============================================")
        self.logger.info("=============================================")

    def build_model_with_adapter_config(self, args, adapter_config):
        """Build model according to adapter configuration

        Args:
            args: Model parameters
            adapter_config: (layer1_count, layer2_count, layer3_count) Number of adapters per layer

        Returns:
            Built model
        """
        # Create base model
        model = build_model(args).to(self.device)

        # If model supports, set Wasserstein distance calculation normalization parameters
        if hasattr(model, "set_wasserstein_params"):
            model.set_wasserstein_params(
                normalize=self.normalize_wasserstein,
                norm_method=self.wasserstein_norm_method,
            )

        if hasattr(model, "use_adapters") and model.use_adapters:
            layer1_count, layer2_count, layer3_count = adapter_config

            # Add adapters to specified count for each layer
            # Layer 1
            current_count_1 = model.adapter_list_1.get_num_adapters()
            for _ in range(layer1_count - current_count_1):
                model.adapter_list_1.add_adapter()

            # Layer 2
            current_count_2 = model.adapter_list_2.get_num_adapters()
            for _ in range(layer2_count - current_count_2):
                model.adapter_list_2.add_adapter()

            # Layer 3
            current_count_3 = model.adapter_list_3.get_num_adapters()
            for _ in range(layer3_count - current_count_3):
                model.adapter_list_3.add_adapter()

            # Ensure newly added adapters are on correct device
            model.to(self.device)

            print(
                f"Built model with adapter config: layer1={layer1_count}, layer2={layer2_count}, layer3={layer3_count}"
            )

            # Verify build result
            actual_config = (
                model.adapter_list_1.get_num_adapters(),
                model.adapter_list_2.get_num_adapters(),
                model.adapter_list_3.get_num_adapters(),
            )
            print(f"Actual adapter counts: {actual_config}")

            if actual_config != adapter_config:
                print(
                    f"Warning: Adapter config mismatch! Expected: {adapter_config}, Got: {actual_config}"
                )

        return model

    def get_current_adapter_config(self):
        """Get adapter configuration of current training model

        Returns:
            tuple: (layer1_count, layer2_count, layer3_count) Number of adapters per layer
        """
        if not (
            hasattr(self.rppg_estimator, "use_adapters")
            and self.rppg_estimator.use_adapters
        ):
            return (1, 1, 1)  # Default configuration

        layer1_count = self.rppg_estimator.adapter_list_1.get_num_adapters()
        layer2_count = self.rppg_estimator.adapter_list_2.get_num_adapters()
        layer3_count = self.rppg_estimator.adapter_list_3.get_num_adapters()

        return (layer1_count, layer2_count, layer3_count)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    ## ! general params.
    parser.add_argument("--num_rppg", type=int, default=160, help="the number of rPPG")
    parser.add_argument(
        "--dataset",
        type=str,
        default="VIPL1to4_BUAA_PURE_UBFC_MMPD_VIPL5",
        help="dataset = [VIPL, UBFC, PURE, COHFACE, VIPL1to4, VIPL5]",
    )
    parser.add_argument(
        "--vipl5_split_ratio", type=float, default=0.8, help="VIPL5 training rate"
    )
    parser.add_argument(
        "--vipl_fold", type=int, default=1, help="the fold of VIPL dataset"
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="PATH_TO_SAVE",
        help="the path to save the model [ckpt, code, visulization]",
    )
    parser.add_argument(
        "--save_mode", type=str, default="all", help="save mode [all, best]"
    )

    ## ! train params.
    parser.add_argument("--gpu", type=str, default="6", help="gpu id list")
    parser.add_argument("--img_size", type=int, default=128, help="the length of clip")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size per gpu")
    parser.add_argument(
        "--eval_step", type=int, default=1, help="the number of **epochs** to eval"
    )
    parser.add_argument(
        "--epochs",
        type=str,
        default="25_15",
        help="the number of epochs to train: inital_other",
    )
    parser.add_argument(
        "--echo_batches",
        type=int,
        default=500,
        help="the number of **mini-batches** to print the loss",
    )
    parser.add_argument(
        "--new_domain_load_mode",
        type=str,
        default="best",
        help="new domain load mode = [best, latest]",
    )
    ### loss
    parser.add_argument(
        "--loss",
        type=str,
        default='["np_loss", "ce_loss", "vae_loss"]',
        help="loss = [np_loss, ce_loss, vae_loss]",
    )
    parser.add_argument(
        "--loss_weight",
        type=str,
        default="[0.2, 1, 0.1]",
        help="loss_weight = [0.2, 1, 0.1]",
    )
    ### eval_option
    parser.add_argument(
        "--eval_type", type=str, default="video", help="eval_type = [clip, video]"
    )

    ## ! model params.
    parser.add_argument(
        "--model", type=str, default="ViT", help="model = [ViT]"
    )
    parser.add_argument("--dropout", type=float, default=0.2, help="dropout rate")
    parser.add_argument(
        "--backbone_freeze_ratio",
        type=float,
        default=0.6,
        help="Proportion of randomly frozen transformer backbone parameters (0.0-1.0)",
    )

    ### optim
    parser.add_argument(
        "--optim", type=str, default="adam", help="optimizer = [adam, sgd]"
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--beta1", type=float, default=0.9, help="beta1 for adam")
    parser.add_argument("--beta2", type=float, default=0.999, help="beta2 for adam")
    parser.add_argument(
        "--weight_decay", type=float, default=5e-5, help="weight decay for optimizer"
    )
    parser.add_argument("--momentum", type=float, default=0.9, help="momentum for sgd")
    ### scheduler
    parser.add_argument(
        "--scheduler", type=str, default="step", help="scheduler = [step]"
    )
    parser.add_argument(
        "--step_size", type=int, default=50, help="learning rate decay step size"
    )
    parser.add_argument("--gamma", type=float, default=0.5, help="learning rate decay")

    # Add intelligent adapter expansion related parameters
    parser.add_argument(
        "--use_intelligent_expand",
        action="store_true",
        default=True,
        help="whether to use intelligent adapter expansion",
    )
    parser.add_argument(
        "--adapter_threshold",
        type=float,
        default=0.05,
        help="threshold for adding new adapters based on Wasserstein distance",
    )
    parser.add_argument(
        "--vae_training_epochs",
        type=int,
        default=5,
        help="number of epochs to save VAE distributions",
    )

    # Add VAE related parameters
    parser.add_argument(
        "--use_vae_loss",
        action="store_true",
        default=True,
        help="whether to use VAE loss",
    )
    parser.add_argument(
        "--vae_kl_weight",
        type=float,
        default=0.001,
        help="KL divergence weight in VAE loss",
    )
    parser.add_argument(
        "--wasserstein_norm",
        type=str,
        default="False",
        choices=["False", "l2", "unit_sphere", "z_score", "minmax"],
        help="normalization method for Wasserstein distance: False (no normalization), l2 (L2 norm), unit_sphere (project to unit sphere), z_score (standard normalization), minmax (min-max scaling)",
    )
    parser.add_argument(
        "--backbone_weight",
        type=float,
        default=0.9,
        help="Weight of backbone main output during fusion (α)",
    )
    parser.add_argument(
        "--adapter_weight",
        type=float,
        default=0.1,
        help="Weight of Adapter output during fusion (β)",
    )

    # Sample annealing (offline precomputation + dynamic Sampler) parameters
    parser.add_argument(
        "--sample_anneal",
        action="store_true",
        default=True,
        help="enable sample curriculum annealing from parent domain (for datasets after the first)",
    )
    parser.add_argument(
        "--sample_anneal_layer",
        type=int,
        default=2,
        choices=[1, 2, 3],
        help="which layer features to compute distances on (pre-FFN)",
    )
    parser.add_argument(
        "--sample_anneal_q0",
        type=float,
        default=0.3,
        help="initial coverage ratio for close-to-parent samples",
    )
    parser.add_argument(
        "--sample_anneal_warmup_epochs",
        type=int,
        default=-1,
        help="epochs to grow coverage to 1.0; -1 means 50% of domain epochs",
    )
    parser.add_argument(
        "--sample_anneal_explore",
        type=float,
        default=0.1,
        help="extra random exploration ratio each epoch",
    )

    # progressive_reg distribution alignment parameters
    parser.add_argument(
        "--progressive_reg_mode",
        type=str,
        default="swd",
        choices=["mse", "swd"],
        help="progressive_reg loss mode: mse or swd",
    )
    parser.add_argument(
        "--progressive_swd_K",
        type=int,
        default=16,
        help="number of directions for sliced Wasserstein",
    )
    parser.add_argument(
        "--progressive_reg_weight",
        type=float,
        default=1.0,
        help="weight for progressive_reg loss",
    )
    # PGP path stability constraint annealing rate (paper's kappa).
    parser.add_argument(
        "--progressive_beta0",
        type=float,
        default=1.0,
        help="initial beta0 for progressive regularization (at t=1)",
    )
    parser.add_argument(
        "--progressive_beta_gamma",
        type=float,
        default=1.0,
        help="constraint decay rate kappa for PGP path stability",
    )

    args = parser.parse_args()

    set_seed(92)

    rppg_estimator_trainer = RppgEstimatorTrainer(args)

    rppg_estimator_trainer.train_continual_learning(
        start_dataset=0, start_epoch=0, continue_log=""
    )
