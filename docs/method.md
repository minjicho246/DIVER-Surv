# Method and outputs

## Representations

Four MRI sequences are loaded in the order T1, T1c, T2, and FLAIR, resized to `120 × 120 × 64`, scaled to `[0, 1]`, and concatenated as a four-channel volume. The timestep-conditioned 3D U-Net predicts injected Gaussian noise and projects its bottleneck representation into four 256-dimensional image tokens.

Structured clinical variables must already be summarized in the metadata CSV. A frozen SapBERT encoder precomputes one text embedding per patient. The fusion model projects the image and text tokens to width 256, prepends a learned virtual node, and applies two Transformer encoder layers with eight attention heads.

## Objective

Each training batch creates two independent diffusion branches. The DDPM term is the ordinary mean squared error between injected and predicted noise:

```text
L_diff = 1/2 Σ_k MSE(epsilon_k, predicted_epsilon_k)
```

Two additional views independently sample timesteps from the inclusive integer range `[0, Ts]`, with `Ts = 100`. Each produces a survival score and contributes a negative Cox partial log-likelihood. A stop-gradient consistency term compares their image embeddings:

```text
L_cons = MSE(z_image(B2), stop_gradient(z_image(B1)))
```

The complete objective is:

```text
L_total = L_surv(B1) + L_surv(B2) + 0.2 L_diff + 0.5 L_cons
```

No ranking loss or Min-SNR weighting is applied.

## Optimization

Adam uses separate parameter groups for the image encoder (`1e-5`) and survival module (`3e-6`) with weight decay `1e-4`. A cosine annealing scheduler runs for 100 epochs. The default batch size is 8 and dropout is 0.2. CUDA automatic mixed precision is enabled by default when CUDA is available.

A warm-up forward pass creates the shape-dependent bottleneck linear layers before constructing Adam, ensuring their parameters belong to the optimizer.

## Evaluation

Every epoch evaluates the validation set, the source-cohort internal test set, and two external cohorts. Survival evaluation uses a clean image input. The median training-split risk score separates high- and low-risk groups.

The code records C-index, log-rank p-value, and its existing high/low event-rate ratio labeled `HR`. Reported bootstrap confidence intervals are outside this training entry point.

## Checkpoints and histories

Each run writes:

- `training_log.txt`
- `results_final_<SITE>.json`
- image and survival state dictionaries for the best validation loss
- image and survival state dictionaries for the best validation C-index
- optional state dictionaries when all three held-out test p-values are below `0.05`

The JSON history contains validation and test C-index, HR, and p-value arrays for every epoch. Optimizer and scheduler states are not saved.
