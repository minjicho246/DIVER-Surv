"""Prepare all cohorts and run the two source-cohort experiments."""

import json
import os
import shutil
from datetime import datetime

import pandas as pd
import torch
import wandb
from monai.data import DataLoader, PersistentDataset
from sklearn.model_selection import train_test_split

from .data.embeddings import precompute_text_embeddings
from .data.indexing import index_rhuh, index_ucsf, index_upenn
from .data.transforms import AugmentedDataset, build_aug_transforms, build_pre_transforms
from .models.diffusion import DiffusionImageEncoder3D
from .models.fusion import VirtualNodeSurvivalMLP
from .training import train_model
from .utils import log_message, safe_wandb_name, to_serializable


def run_experiments(args, runtime):
    """Execute cohort setup, source-site training, result saving, and cache cleanup."""

    # Resolve process settings and build the experiment name.
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = args.batch_size
    NUM_WORKERS = max(0, args.num_workers)
    PREFETCH_FACTOR = max(2, args.prefetch_factor)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    UCSF_CSV_PATH = runtime.ucsf_csv_path
    UCSF_IMG_DIR = runtime.ucsf_img_dir
    UPENN_CSV_PATH = runtime.upenn_csv_path
    UPENN_IMG_DIR = runtime.upenn_img_dir
    RHUH_CSV_PATH = runtime.rhuh_csv_path
    RHUH_IMG_DIR = runtime.rhuh_img_dir

    exp_name = (
        f"seed{args.seed1}_seed{args.seed2}"
        f"_lr1_{args.lr1}_lr2_{args.lr2}"
        f"_bs_{BATCH_SIZE}"
        f"_ep_{args.epochs}"
        f"_textmodel_{args.text_model}"
        f"_norm_{args.norm_type}"
        f"_g{args.gn_groups}"
        f"_drop_{args.dropout_rate}"
        f"_diffstep_{args.diffusion_steps}"
        f"_dbeta_{args.diffusion_beta_start}_{args.diffusion_beta_end}"
        f"_tdim_{args.diffusion_time_embed_dim}"
        f"_diffemb_{args.diffusion_embed_dim}"
        f"_survh_{args.survival_hidden_dim}"
        f"_stmax_{args.survival_t_max}"
        f"_ldiff_{args.lambda_diff}"
        f"_lcons_{args.consi_loss}"
        f"_wd_{args.weight_decay}"
        f"_heads_{args.fusion_num_heads}"
    )
    BASE_OUTPUT_DIR = runtime.output_root + f"/{exp_name}"

    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

    ucsf_df = pd.read_csv(UCSF_CSV_PATH)
    upenn_df = pd.read_csv(UPENN_CSV_PATH)
    rhuh_df = pd.read_csv(RHUH_CSV_PATH)

    # Encode every cohort once before either source-site experiment starts.
    print("Pre-computing text embeddings...")
    ucsf_emb, text_dim = precompute_text_embeddings(
        ucsf_df, "Patient_Summary", device=DEVICE, model_name=args.text_model
    )
    upenn_emb, _ = precompute_text_embeddings(
        upenn_df, "Patient_Summary", device=DEVICE, model_name=args.text_model
    )
    rhuh_emb, _ = precompute_text_embeddings(
        rhuh_df, "Patient_Summary", device=DEVICE, model_name=args.text_model
    )

    # Run both source-cohort settings with a consistent modality order.
    experiments = [("UPENN", ["UCSF", "RHUH"]), ("UCSF", ["UPENN", "RHUH"])]

    KEYS = {
        "UCSF": ["T1_bias", "T1c_bias", "T2_bias", "FLAIR_bias"],
        "UPENN": ["T1.nii.gz", "T1GD.nii.gz", "T2.nii.gz", "FLAIR.nii.gz"],
        "RHUH": ["t1", "t1ce", "t2", "flair"],
    }

    PRE_TFS = {site: build_pre_transforms(KEYS[site]) for site in KEYS}
    train_aug = build_aug_transforms(train=True)
    test_aug = build_aug_transforms(train=False)
    group_name = safe_wandb_name(exp_name, max_len=120)
    for source_site, target_sites in experiments:
        print(f"\n🔥 === Starting Experiment: Train on {source_site} ===")
        log_message(f"\n🔥 === Starting Experiment: Train on {source_site} ===", None)
        run_name = safe_wandb_name(f"{group_name}_{source_site}", max_len=120)
        run = wandb.init(
            entity=runtime.wandb_entity,
            project=runtime.wandb_project,
            name=run_name,
            group=group_name,
            reinit="finish_previous",
            config={
                "seed1": args.seed1,
                "seed2": args.seed2,
                "text_model": args.text_model,
                "learning_rate1": args.lr1,
                "learning_rate2": args.lr2,
                "weight_decay": args.weight_decay,
                "architecture": "dual-branch diffusion + small-t survival + virtual-node fusion",
                "dataset": "UCSF + UPenn + RHUH",
                "source_site": source_site,
                "target_sites": target_sites,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "num_workers": NUM_WORKERS,
                "prefetch_factor": PREFETCH_FACTOR if NUM_WORKERS > 0 else 0,
                "dropout_rate": args.dropout_rate,
                "norm_type": args.norm_type,
                "gn_groups": args.gn_groups,
                "eta_min": args.eta_min,
                "consi_loss": args.consi_loss,
                "use_amp": args.use_amp,
                "lambda_diff": args.lambda_diff,
                "diffusion_steps": args.diffusion_steps,
                "diffusion_beta_start": args.diffusion_beta_start,
                "diffusion_beta_end": args.diffusion_beta_end,
                "diffusion_time_embed_dim": args.diffusion_time_embed_dim,
                "diffusion_embed_dim": args.diffusion_embed_dim,
                "survival_hidden_dim": args.survival_hidden_dim,
                "fusion_num_layers": args.fusion_num_layers,
                "fusion_num_heads": args.fusion_num_heads,
                "survival_t_max": args.survival_t_max,
                "code_version": "public-paper-release",
            },
        )

        exp_name = f"source_{source_site}_seed{args.seed1}_{args.seed2}"
        output_dir = os.path.join(BASE_OUTPUT_DIR, f"{timestamp}_{exp_name}")
        CACHE_DIR = f"{runtime.cache_root}/{timestamp}_{exp_name}"
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(CACHE_DIR, exist_ok=True)
        print(f"Output Directory: {output_dir}")
        print(f"Cache Directory: {CACHE_DIR}")
        if not os.path.exists(CACHE_DIR):
            raise FileNotFoundError("Cache directory was not created successfully.")
        if not os.path.exists(output_dir):
            raise FileNotFoundError("Output directory was not created successfully.")
        log_file = os.path.join(output_dir, "training_log.txt")

        # Preserve source-specific seeds and the two-stage stratified split.
        if source_site == "UPENN":
            src_df, src_emb, src_idx_fn, src_img_dir = (
                upenn_df,
                upenn_emb,
                index_upenn,
                UPENN_IMG_DIR,
            )
            stratify_col = "event"
            tr_idx, temp_idx = train_test_split(
                src_df.index, test_size=0.3, stratify=src_df[stratify_col], random_state=args.seed1
            )
            val_idx, te_idx = train_test_split(
                temp_idx,
                test_size=2 / 3,
                stratify=src_df.loc[temp_idx, stratify_col],
                random_state=args.seed1,
            )

        else:
            src_df, src_emb, src_idx_fn, src_img_dir = ucsf_df, ucsf_emb, index_ucsf, UCSF_IMG_DIR
            stratify_col = "1-dead 0-alive"
            tr_idx, temp_idx = train_test_split(
                src_df.index, test_size=0.3, stratify=src_df[stratify_col], random_state=args.seed2
            )
            val_idx, te_idx = train_test_split(
                temp_idx,
                test_size=2 / 3,
                stratify=src_df.loc[temp_idx, stratify_col],
                random_state=args.seed2,
            )

        # Cache deterministic preprocessing; apply augmentation per item access.
        def get_loader(items, site, aug, shuffle=False):
            ds = AugmentedDataset(PersistentDataset(items, PRE_TFS[site], cache_dir=CACHE_DIR), aug)
            loader_kwargs = dict(
                batch_size=BATCH_SIZE,
                num_workers=NUM_WORKERS,
                shuffle=shuffle,
                pin_memory=True,
            )
            if NUM_WORKERS > 0:
                loader_kwargs["persistent_workers"] = True
                loader_kwargs["prefetch_factor"] = PREFETCH_FACTOR
            return DataLoader(ds, **loader_kwargs)

        tr_loader = get_loader(
            src_idx_fn(src_df.loc[tr_idx], src_img_dir, src_emb[tr_idx]),
            source_site,
            train_aug,
            True,
        )
        val_loader = get_loader(
            src_idx_fn(src_df.loc[val_idx], src_img_dir, src_emb[val_idx]), source_site, test_aug
        )
        internal_te_loader = get_loader(
            src_idx_fn(src_df.loc[te_idx], src_img_dir, src_emb[te_idx]), source_site, test_aug
        )

        # Both external cohorts are evaluated alongside the internal test split.
        external_loaders = {}
        for target in target_sites:
            if target == "UCSF":
                items = index_ucsf(ucsf_df, UCSF_IMG_DIR, ucsf_emb)
            elif target == "UPENN":
                items = index_upenn(upenn_df, UPENN_IMG_DIR, upenn_emb)
            elif target == "RHUH":
                items = index_rhuh(rhuh_df, RHUH_IMG_DIR, rhuh_emb)

            external_loaders[target] = get_loader(items, target, test_aug)

        # Construct the paper architecture.
        img_model = DiffusionImageEncoder3D(
            in_channels=4,
            latent_dim=128,
            embed_dim=args.diffusion_embed_dim,
            dropout_rate=args.dropout_rate,
            norm_type=args.norm_type,
            gn_groups=args.gn_groups,
            diffusion_steps=args.diffusion_steps,
            beta_start=args.diffusion_beta_start,
            beta_end=args.diffusion_beta_end,
            time_embed_dim=args.diffusion_time_embed_dim,
        )
        surv_model = VirtualNodeSurvivalMLP(
            image_dim=args.diffusion_embed_dim,
            text_dim=text_dim,
            hidden_dim=args.survival_hidden_dim,
            dropout_rate=args.dropout_rate,
            num_layers=args.fusion_num_layers,
            num_heads=args.fusion_num_heads,
        )

        all_test_loaders = {f"Internal_{source_site}": internal_te_loader, **external_loaders}

        results = train_model(
            img_model,
            surv_model,
            (tr_loader, val_loader, all_test_loaders),
            num_epochs=args.epochs,
            lr=args.lr1,
            lr2=args.lr2,
            device=DEVICE,
            log_file=log_file,
            output_dir=output_dir,
            source_site=source_site,
            eta_min=args.eta_min,
            args=args,
        )

        # Persist histories and remove the completed experiment's cache.
        with open(os.path.join(output_dir, f"results_final_{source_site}.json"), "w") as f:
            json.dump(to_serializable(results), f, indent=2)
        run.finish()
        shutil.rmtree(CACHE_DIR)
        print(f"✅ Finished {source_site} Experiment. Cache removed.")
