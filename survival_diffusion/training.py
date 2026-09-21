"""Training, evaluation, logging, and checkpoint orchestration."""

import os
from contextlib import nullcontext

import torch
import torch.nn.functional as F
import torch.optim as optim
import wandb

from .losses.diffusion import diffusion_loss
from .losses.survival import CoxPHLoss
from .metrics import calc_metrics
from .utils import log_message


def train_model(
    imgmodel,
    survmodel,
    loaders,
    num_epochs,
    lr,
    lr2,
    device,
    log_file,
    output_dir,
    source_site,
    eta_min=1e-7,
    *,
    args,
):
    """Train DIVER-Surv and evaluate the validation and held-out cohorts each epoch."""
    train_loader, val_loader, test_loaders = loaders
    imgmodel.to(device)
    survmodel.to(device)

    # The bottleneck FC dimensions depend on the resized image shape.
    with torch.no_grad():
        warm_batch = next(iter(train_loader))
        warm_imgs = warm_batch["image"].to(device, non_blocking=True)
        imgmodel(warm_imgs[:1], add_noise=False)

    optimizer = optim.Adam(
        [
            {"params": imgmodel.parameters(), "lr": lr},
            {"params": survmodel.parameters(), "lr": lr2},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=eta_min)
    use_amp = bool(args.use_amp) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=True) if use_amp else None
    survival_loss = CoxPHLoss()
    survival_t_max = max(0, min(args.survival_t_max, args.diffusion_steps - 1))

    history = {
        key: [0.0] * num_epochs
        for key in ["train_loss", "val_loss", "train_c", "val_c", "median_risk"]
    }
    test_history = {
        site: {metric: [0.0] * num_epochs for metric in ["c", "hr", "p"]}
        for site in ["Validation", *test_loaders]
    }
    best_val_loss = float("inf")
    best_val_c = 0.0

    def encode_survival_view(images):
        timestep = imgmodel.sample_timesteps(images.size(0), images.device, t_max=survival_t_max)
        return imgmodel(images, add_noise=True, t_override=timestep)

    def evaluate(loader):
        imgmodel.eval()
        survmodel.eval()
        survival_sum = 0.0
        diffusion_sum = 0.0
        times, risks, events = [], [], []

        with torch.no_grad():
            for batch in loader:
                images = batch["image"].to(device, non_blocking=True)
                texts = batch["text_emb"].to(device, non_blocking=True)
                batch_events = batch["event"].to(device, non_blocking=True)
                batch_times = batch["time"].to(device, non_blocking=True)

                amp_context = torch.amp.autocast("cuda", enabled=True) if use_amp else nullcontext()
                with amp_context:
                    clean_output = imgmodel(images, add_noise=False)
                    scores, _ = survmodel(clean_output["img_emb"], texts)
                    noisy_output = imgmodel(images, add_noise=True)
                    batch_diffusion_loss = diffusion_loss(
                        noisy_output["pred_noise"], noisy_output["target_noise"]
                    )
                batch_survival_loss = survival_loss(scores.float(), batch_events, batch_times)

                survival_sum += batch_survival_loss.item()
                diffusion_sum += batch_diffusion_loss.item()
                times.extend(batch_times.detach().cpu().tolist())
                risks.extend(scores.detach().cpu().reshape(-1).tolist())
                events.extend(batch_events.detach().cpu().tolist())

        num_batches = max(1, len(loader))
        return survival_sum / num_batches, diffusion_sum / num_batches, times, risks, events

    for epoch in range(num_epochs):
        imgmodel.train()
        survmodel.train()
        total_loss = 0.0
        total_survival = 0.0
        total_diffusion = 0.0
        total_consistency = 0.0
        train_times, train_risks, train_events = [], [], []

        for batch in train_loader:
            images = batch["image"].to(device)
            texts = batch["text_emb"].to(device)
            events = batch["event"].to(device)
            times = batch["time"].to(device)
            optimizer.zero_grad(set_to_none=True)

            amp_context = torch.amp.autocast("cuda", enabled=True) if use_amp else nullcontext()
            with amp_context:
                # Independent DDPM branches use separate timesteps and noise draws.
                noisy_branch_1 = imgmodel(images, add_noise=True)
                noisy_branch_2 = imgmodel(images, add_noise=True)
                diffusion_branch_1 = diffusion_loss(
                    noisy_branch_1["pred_noise"], noisy_branch_1["target_noise"]
                )
                diffusion_branch_2 = diffusion_loss(
                    noisy_branch_2["pred_noise"], noisy_branch_2["target_noise"]
                )
                batch_diffusion = 0.5 * (diffusion_branch_1 + diffusion_branch_2)

                # Survival views sample from the paper's small timestep range [0, Ts].
                survival_branch_1 = encode_survival_view(images)
                survival_branch_2 = encode_survival_view(images)
                scores_1, _ = survmodel(survival_branch_1["img_emb"], texts)
                scores_2, _ = survmodel(survival_branch_2["img_emb"], texts)
                batch_consistency = F.mse_loss(
                    survival_branch_2["img_emb"],
                    survival_branch_1["img_emb"].detach(),
                )

            batch_survival = survival_loss(scores_1.float(), events, times) + survival_loss(
                scores_2.float(), events, times
            )
            loss = (
                batch_survival
                + args.lambda_diff * batch_diffusion.float()
                + args.consi_loss * batch_consistency.float()
            )

            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            total_survival += batch_survival.item()
            total_diffusion += batch_diffusion.item()
            total_consistency += batch_consistency.item()
            train_times.extend(times.detach().cpu().tolist())
            train_risks.extend(scores_1.detach().cpu().reshape(-1).tolist())
            train_events.extend(events.detach().cpu().tolist())

        scheduler.step()
        num_train_batches = max(1, len(train_loader))
        train_loss = total_loss / num_train_batches
        train_survival = total_survival / num_train_batches
        train_diffusion = total_diffusion / num_train_batches
        train_consistency = total_consistency / num_train_batches
        train_c, train_p, train_hr, median_risk = calc_metrics(
            train_times, train_risks, train_events
        )

        val_survival, val_diffusion, val_times, val_risks, val_events = evaluate(val_loader)
        val_c, val_p, val_hr, _ = calc_metrics(val_times, val_risks, val_events, median_risk)
        val_result = {
            "loss": val_survival,
            "objective": val_survival + args.lambda_diff * val_diffusion,
            "surv": val_survival,
            "diff": val_diffusion,
            "c": val_c,
            "hr": val_hr,
            "p": val_p,
        }

        current_test_results = {}
        for site_name, loader in test_loaders.items():
            test_survival, test_diffusion, test_times, test_risks, test_events = evaluate(loader)
            test_c, test_p, test_hr, _ = calc_metrics(
                test_times, test_risks, test_events, median_risk
            )
            current_test_results[site_name] = {
                "loss": test_survival,
                "objective": test_survival + args.lambda_diff * test_diffusion,
                "surv": test_survival,
                "diff": test_diffusion,
                "c": test_c,
                "hr": test_hr,
                "p": test_p,
            }

        history["train_loss"][epoch] = train_loss
        history["val_loss"][epoch] = val_result["loss"]
        history["train_c"][epoch] = train_c
        history["val_c"][epoch] = val_result["c"]
        history["median_risk"][epoch] = median_risk
        for metric in ["c", "hr", "p"]:
            test_history["Validation"][metric][epoch] = val_result[metric]
            for site_name, result in current_test_results.items():
                test_history[site_name][metric][epoch] = result[metric]

        log_data = {
            "epoch": epoch + 1,
            "train/loss": train_loss,
            "train/surv": train_survival,
            "train/diff": train_diffusion,
            "train/consistency": train_consistency,
            "train/c_index": train_c,
            "train/hr": train_hr,
            "train/p_value": train_p,
            "val/loss": val_result["loss"],
            "val/objective": val_result["objective"],
            "val/surv": val_result["surv"],
            "val/diff": val_result["diff"],
            "val/c_index": val_result["c"],
            "val/hr": val_result["hr"],
            "val/p_value": val_result["p"],
            "median_risk": median_risk,
        }
        for site_name, result in current_test_results.items():
            for metric, value in result.items():
                log_data[f"test_{site_name}/{metric}"] = value
        wandb.log(log_data)

        log_message(
            f"Ep {epoch + 1} | Tr_L: {train_loss:.3f} | Tr_C: {train_c:.3f} "
            f"| Med_Risk: {median_risk:.3f} | Tr_HR: {train_hr:.2f} "
            f"| Tr_P: {train_p:.4f} | Tr_Surv: {train_survival:.3f} "
            f"| Tr_Diff: {train_diffusion:.3f} | Val_L: {val_result['loss']:.3f} "
            f"| Val_C: {val_result['c']:.3f} | Val_HR: {val_result['hr']:.2f} "
            f"| Val_P: {val_result['p']:.4f}",
            log_file,
        )
        for site_name, result in current_test_results.items():
            log_message(
                f"   [{site_name}] C: {result['c']:.3f} | HR: {result['hr']:.2f} "
                f"| P: {result['p']:.4f}",
                log_file,
            )

        epoch_tag = f"epoch{epoch + 1}"
        test_p_values = [result["p"] for result in current_test_results.values()]
        if len(test_p_values) == 3 and all(0.0 <= p < 0.05 for p in test_p_values):
            torch.save(
                imgmodel.state_dict(),
                os.path.join(
                    output_dir,
                    f"{source_site}_alltest_p_lt_0.05_{epoch_tag}_diffimgmodel.pth",
                ),
            )
            torch.save(
                survmodel.state_dict(),
                os.path.join(
                    output_dir,
                    f"{source_site}_alltest_p_lt_0.05_{epoch_tag}_survmodel.pth",
                ),
            )

        if val_result["loss"] < best_val_loss:
            best_val_loss = val_result["loss"]
            torch.save(
                imgmodel.state_dict(),
                os.path.join(output_dir, f"{source_site}_best_diffimgmodel_{epoch_tag}.pth"),
            )
            torch.save(
                survmodel.state_dict(),
                os.path.join(output_dir, f"{source_site}_best_survmodel_{epoch_tag}.pth"),
            )
            log_message(f"   --> Best validation loss: {val_result['loss']:.4f}", log_file)

        if val_result["c"] > best_val_c:
            best_val_c = val_result["c"]
            torch.save(
                imgmodel.state_dict(),
                os.path.join(output_dir, f"{source_site}_best_c_diffimgmodel_{epoch_tag}.pth"),
            )
            torch.save(
                survmodel.state_dict(),
                os.path.join(output_dir, f"{source_site}_best_c_survmodel_{epoch_tag}.pth"),
            )
            log_message(f"   --> Best validation C-index: {val_result['c']:.4f}", log_file)

    history["test_sites"] = test_history
    return history
