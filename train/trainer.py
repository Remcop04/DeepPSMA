import torch
from torch import optim
from torch.amp import GradScaler
from monai.losses import DiceLoss, DiceCELoss
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import Activations, AsDiscrete
from monai.utils import MetricReduction
from torchinfo import summary

import time
import json

def log_model(run, model, device, example_input_size=(2, 128, 128, 128)):
    try:
        # Gedetailleerde laag-per-laag summary via torchinfo
        model_summary_str = str(summary(model, input_size=example_input_size, device=str(device)))
    except Exception:
        # Fallback: alleen de simpele repr
        model_summary_str = repr(model)
    run.log({"model_summary": model_summary_str})

def log_config(run, cfg, loss_function, optimizer):
    # Log alle hyperparameters en relevante configuratie in wandb config
    run.config.update({
        "learning_rate": cfg.get("lr"),
        "batch_size": cfg.get("batch_size"),
        "epochs": cfg.get("epochs"),
        "loss_function": str(loss_function),
        "optimizer": str(optimizer),
        # Voeg hier alles toe wat je relevant vindt
    })

def train(cfg, model, train_loader, val_loader, device, run):

    # === SETUP ===
    loss_function = DiceCELoss(to_onehot_y=False, sigmoid=True, lambda_dice=2.0, lambda_ce=0.5) #DiceLoss(to_onehot_y=False, sigmoid=True)
    optimizer = optim.Adam(model.parameters(), lr=cfg["lr"])
    scaler = GradScaler()

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=5,
        min_lr=1e-7
    )

    # Log info bij start
    log_config(run, cfg, loss_function, optimizer)
    log_model(run, model, device, example_input_size=(1, 2) + cfg["image_size"])

    post_pred = Activations(sigmoid=True)
    post_label = AsDiscrete(threshold=0.5)
    dice_metric = DiceMetric(include_background=False, reduction=MetricReduction.MEAN)  # adviezen toegepast

    best_val_dice = -1.0

    for epoch in range(cfg["epochs"]):
        model.train()
        torch.cuda.reset_peak_memory_stats(device)
        epoch_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            start_time = time.time()

            pet = batch["pet"].to(device)
            ct = batch["ct"].to(device)
            labels = batch["mask"].to(device)
            inputs = torch.cat([pet, ct], dim=1)

            pos_voxels = (labels > 0).sum().item()
            total_voxels = labels.numel()
            positive_ratio = pos_voxels / total_voxels
            run.log({"positive_voxel_ratio": positive_ratio})

            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                loss = loss_function(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

            batch_time = time.time() - start_time

            if batch_idx % 10 == 0:
                # Leer tempo en geheugen loggen
                current_lr = optimizer.param_groups[0]['lr']
                gpu_mem_allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)  # GB
                gpu_mem_reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)    # GB

                run.log({
                    "batch_loss": loss.item(),
                    "epoch": epoch + 1,
                    "batch": batch_idx,
                    "learning_rate": current_lr,
                    "gpu_mem_allocated_GB": gpu_mem_allocated,
                    "gpu_mem_reserved_GB": gpu_mem_reserved,
                    "batch_duration_sec": batch_time
                })

        avg_train_loss = epoch_loss / len(train_loader)
        print(f"[Epoch {epoch+1}] Train Loss: {avg_train_loss:.4f}")
        run.log({"epoch_train_loss": avg_train_loss, "epoch": epoch + 1})

        peak_allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        run.log({"gpu_mem_peak_allocated_GB": peak_allocated})

        # === VALIDATION ===
        model.eval()
        dice_metric.reset()
        val_loss = 0.0

        with torch.no_grad():
            for val_batch in val_loader:
                pet = val_batch["pet"].to(device)
                ct = val_batch["ct"].to(device)
                labels = val_batch["mask"].to(device)
                inputs = torch.cat([pet, ct], dim=1)

                with torch.amp.autocast('cuda'):
                    outputs = sliding_window_inference(
                        inputs,
                        roi_size=cfg["image_size"],
                        sw_batch_size=cfg.get("batch_size", 4),
                        predictor=model,
                        overlap=0.5
                    )
                    loss = loss_function(outputs, labels)
                    val_loss += loss.item()

                    preds = post_pred(outputs)
                    labels_post = post_label(labels)
                    dice_metric(preds, labels_post)

        # === LOGGEN VAN VOORSPELLINGEN NAAR WANDB ===
        if batch_idx == 0:  # Alleen van eerste batch
            max_slices = 5  # log max 3 slices
            images_to_log = []

            for j in range(min(pet.shape[0], max_slices)):
                # Kies centrale slice
                center_slice = pet[j, 0].shape[2] // 2

                pet_slice = pet[j, 0, :, :, center_slice].detach().cpu()
                ct_slice = ct[j, 0, :, :, center_slice].detach().cpu()
                label_slice = labels[j, 0, :, :, center_slice].detach().cpu()
                pred_slice = preds[j, 0, :, :, center_slice].detach().cpu()

                image = wandb.Image(
                    pet_slice,
                    caption=f"Epoch {epoch+1} | Sample {j+1} | PET\nGT (red) / Pred (green)",
                    masks={
                        "ground_truth": {
                            "mask_data": label_slice,
                            "class_labels": {1: "Lesion"},
                        },
                        "prediction": {
                            "mask_data": pred_slice > 0.5,
                            "class_labels": {1: "Pred"},
                        },
                    }
                )
                images_to_log.append(image)

            run.log({f"val_samples_epoch_{epoch+1}": images_to_log})

        avg_val_loss = val_loss / len(val_loader)
        avg_dice = dice_metric.aggregate().item()
        print(f"[Epoch {epoch+1}] Val Loss: {avg_val_loss:.4f} | Dice: {avg_dice:.4f}")

        run.log({
            "epoch_val_loss": avg_val_loss,
            "epoch_val_dice": avg_dice,
            "epoch": epoch + 1
        })

        scheduler.step(avg_dice)

        if avg_dice > best_val_dice:
            best_val_dice = avg_dice
            torch.save(model.state_dict(), "best_model.pth")
            run.save("best_model.pth")

    torch.save(model.state_dict(), "final_model.pth")
    run.save("final_model.pth")
    run.log({"training_completed": True})
    run.finish()
