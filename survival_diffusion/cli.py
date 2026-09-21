"""Command-line options and environment setup."""

import argparse
import os

from .settings import load_runtime_config


def build_parser():
    """Build the training CLI with the paper configuration as defaults."""
    parser = argparse.ArgumentParser(
        description="Train DIVER-Surv on the UPENN and UCSF source cohorts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", default=None, help="JSON file overriding filesystem and W&B settings."
    )

    optimization = parser.add_argument_group("Optimization and data loading")
    optimization.add_argument(
        "--lr1", type=float, default=1e-5, help="Image encoder learning rate."
    )
    optimization.add_argument(
        "--lr2", type=float, default=3e-6, help="Survival model learning rate."
    )
    optimization.add_argument("--weight_decay", type=float, default=1e-4, help="Adam weight decay.")
    optimization.add_argument("--batch_size", type=int, default=8, help="Patients per batch.")
    optimization.add_argument("--epochs", type=int, default=100, help="Epochs per source cohort.")
    optimization.add_argument("--num_workers", type=int, default=4, help="Data-loader workers.")
    optimization.add_argument(
        "--prefetch_factor", type=int, default=2, help="Batches prefetched per worker."
    )
    optimization.add_argument(
        "--eta_min", type=float, default=1e-7, help="Minimum cosine-annealing learning rate."
    )
    optimization.add_argument("--cuda_device", default=None, help="CUDA_VISIBLE_DEVICES value.")
    optimization.add_argument(
        "--use_amp", type=int, default=1, help="Enable CUDA mixed precision when nonzero."
    )

    architecture = parser.add_argument_group("Architecture and text encoder")
    architecture.add_argument("--dropout_rate", type=float, default=0.2, help="Model dropout.")
    architecture.add_argument(
        "--norm_type",
        default="group",
        choices=["group", "instance", "batch"],
        help="3D convolution normalization.",
    )
    architecture.add_argument("--gn_groups", type=int, default=8, help="GroupNorm groups.")
    architecture.add_argument(
        "--text_model",
        default="cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
        help="Frozen SentenceTransformer model name or local path.",
    )
    architecture.add_argument(
        "--diffusion_embed_dim", type=int, default=256, help="Image-node embedding dimension."
    )
    architecture.add_argument(
        "--survival_hidden_dim", type=int, default=256, help="Fusion hidden dimension."
    )
    architecture.add_argument(
        "--fusion_num_layers", type=int, default=2, help="Transformer encoder layers."
    )
    architecture.add_argument(
        "--fusion_num_heads", type=int, default=8, help="Attention heads per transformer layer."
    )

    objective = parser.add_argument_group("Paper objective")
    objective.add_argument(
        "--lambda_diff", type=float, default=0.2, help="Diffusion-loss coefficient."
    )
    objective.add_argument(
        "--consi_loss", type=float, default=0.5, help="Stop-gradient consistency coefficient."
    )
    objective.add_argument(
        "--survival_t_max",
        type=int,
        default=100,
        help="Maximum diffusion timestep sampled for survival views.",
    )

    diffusion = parser.add_argument_group("Diffusion")
    diffusion.add_argument(
        "--diffusion_steps", type=int, default=1000, help="Number of diffusion timesteps."
    )
    diffusion.add_argument(
        "--diffusion_beta_start", type=float, default=1e-4, help="First linear-schedule beta."
    )
    diffusion.add_argument(
        "--diffusion_beta_end", type=float, default=0.02, help="Last linear-schedule beta."
    )
    diffusion.add_argument(
        "--diffusion_time_embed_dim",
        type=int,
        default=128,
        help="Timestep embedding dimension.",
    )

    splits = parser.add_argument_group("Data split seeds")
    splits.add_argument("--seed1", type=int, default=42, help="UPENN split seed.")
    splits.add_argument("--seed2", type=int, default=763, help="UCSF split seed.")
    return parser


def configure_environment(args):
    """Set device and tokenizer variables before importing the ML stack."""
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if args.cuda_device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main(argv=None):
    """Parse options and run both source-cohort experiments."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        runtime = load_runtime_config(args.config)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    configure_environment(args)

    from .experiment import run_experiments

    run_experiments(args, runtime)
