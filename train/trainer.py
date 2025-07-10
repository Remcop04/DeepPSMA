import torch
import torch.nn.functional as F

from torch import optim
from torch.amp import GradScaler
from monai.losses import DiceLoss, DiceCELoss
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric, ConfusionMatrixMetric
from monai.transforms import Activations, AsDiscrete
from monai.utils import MetricReduction
from torchinfo import summary

import wandb
import time
import json

class SensitivityDiceLoss(torch.nn.Module):
    def __init__(self, weight_sensitivity=0.9, weight_dice=0.1, smooth=1e-6):
        super().__init__()
        self.weight_sensitivity = weight_sensitivity
        self.weight_dice = weight_dice
        self.smooth = smooth

    def forward(self, y_pred, y_true):
        y_pred = F.sigmoid(y_pred)

        # Ensure y_true is long and shape [B, H, W, D]
        if y_true.dim() == 5 and y_true.shape[1] == 1:
            y_true = y_true.squeeze(1)

        y_true_onehot = F.one_hot(y_true.long(), num_classes=y_pred.shape[1]).permute(0, 4, 1, 2, 3).float()

        sensitivity_loss = 0.0
        dice_loss = 0.0
        num_classes = y_pred.shape[1]

        for c in range(num_classes):
            p = y_pred[:, c]
            t = y_true_onehot[:, c]

            pred_bin = torch.round(p)

            tp = torch.sum(t * pred_bin)
            fn = torch.sum(t * (1 - pred_bin))
            sensitivity_c = tp / (tp + fn + self.smooth)
            sensitivity_loss += (1 - sensitivity_c)

            intersection = torch.sum(t * p)
            union = torch.sum(t) + torch.sum(p)
            dice_c = (2 * intersection + self.smooth) / (union + self.smooth)
            dice_loss += (1 - dice_c)

        sensitivity_loss /= num_classes
        dice_loss /= num_classes

        total_loss = self.weight_sensitivity * sensitivity_loss + self.weight_dice * dice_loss
        return total_loss

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
    #loss_function = DiceLoss(to_onehot_y=True, softmax=True)
    loss_function = DiceCELoss(to_onehot_y=True, softmax=True, lambda_dice=0.8, lambda_ce=0.2)
    #loss_function = SensitivityDiceLoss(weight_sensitivity=0.9, weight_dice=0.1)
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

    best_val_dice = -1.0

    dice_metric_bin_train = DiceMetric(include_background=False, reduction=MetricReduction.MEAN)
    sens_metric_bin_train = ConfusionMatrixMetric(include_background=False, metric_name='sensitivity', reduction=MetricReduction.MEAN)

    tracer = cfg['tracer']
    for epoch in range(cfg["epochs"]):
        model.train()
        torch.cuda.reset_peak_memory_stats(device)
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            start_time = time.time()
            
            pet = batch["pet"].to(device)
            pet_mask = batch["pet_copy"].to(device)
            ct = batch["ct"].to(device)
            labels = batch["mask"].to(device)
            #inputs = torch.cat([pet, ct], dim=1)
            inputs = torch.cat([pet, ct, pet_mask], dim=1)

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

            # outputs: [B, C, H, W, D]
            preds = torch.argmax(outputs, dim=1)
            y_pred_class1 = (preds == 1).to(torch.float32).unsqueeze(1)
            y_true_class1 = (labels == 1).to(torch.float32)
            
            dice_metric_bin_train(y_pred_class1, y_true_class1)
            sens_metric_bin_train(y_pred_class1, y_true_class1)
            
            num_batches += 1

        avg_train_loss = epoch_loss / len(train_loader)
        avg_dice = dice_metric_bin_train.aggregate().item()
        avg_sensitivity = sens_metric_bin_train.aggregate()[0].item()
        print(f"[Epoch {epoch+1}] Train Loss: {avg_train_loss:.4f}")
        run.log({
                "epoch_train_loss": avg_train_loss,
                "epoch": epoch + 1,
                "epoch_train_dice": avg_dice,
                "epoch_train_sensitivity": avg_sensitivity})

        dice_metric_bin_train.reset()
        sens_metric_bin_train.reset()
        peak_allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        run.log({"gpu_mem_peak_allocated_GB": peak_allocated})

        # === VALIDATION ===
        model.eval() #collate_fn=pad_list_data_collate
        val_loss = 0.0
        with torch.no_grad():
            dice_metric_bin_val = DiceMetric(include_background=False, reduction=MetricReduction.MEAN)
            sens_metric_bin_val = ConfusionMatrixMetric(include_background=False, metric_name='sensitivity', reduction=MetricReduction.MEAN)
            for val_batch_idx, val_batch in enumerate(val_loader):
                
                pet = val_batch["pet"].to(device)
                pet_mask = val_batch["pet_copy"].to(device)
                ct = val_batch["ct"].to(device)
                labels = val_batch["mask"].to(device)
                inputs = torch.cat([pet, ct, pet_mask], dim=1)
                #inputs = torch.cat([pet, ct], dim=1)
                
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

                    preds = torch.argmax(outputs, dim=1)
                    y_pred_class1 = (preds == 1).to(torch.float32).unsqueeze(1)
                    y_true_class1 = (labels == 1).to(torch.float32)

                    num_batches += 1

                    dice_metric_bin_val(y_pred_class1, y_true_class1)
                    sens_metric_bin_val(y_pred_class1, y_true_class1)

                # === LOGGEN VAN VOORSPELLINGEN NAAR WANDB ===
                max_slices = 5
                label_sums = []
                slice_indices = []
                for j in range(pet.shape[0]):
                    center_slice = pet[j, 0].shape[2] // 2
                    label_slice = (labels[j] == 1).float()[0, :, :, center_slice]  # [C,H,W,D]
                    label_sum = torch.sum(label_slice).item()
                    label_sums.append(label_sum)
                    slice_indices.append((j, center_slice))

                # Sorteer en pak top-n
                top_indices = [idx for _, idx in sorted(zip(label_sums, slice_indices), reverse=True) if _ > 0][:max_slices]

                for (j, s) in top_indices:
                    label_slice = (labels[j] == 1).float().squeeze(0)[:, :, center_slice]  # shape: [H, W]
                    pet_slice = pet[j, 0, :, :, center_slice].detach().cpu()               # shape: [H, W]
                    pred_slice = torch.argmax(outputs, dim=1)[j, :, :, center_slice].detach().cpu()  # shape: [H, W]

                    run.log({
                        f"val/pet_top_sample{j}": wandb.Image(pet_slice.cpu(), caption="PET"),
                        f"val/gt_class1_top_sample{j}": wandb.Image(label_slice.cpu(), caption="GT"),
                        f"val/pred_top_sample{j}": wandb.Image(pred_slice.cpu(), caption="Pred"),
                    })
                        
        avg_val_loss = val_loss / len(val_loader)
        avg_dice = dice_metric_bin_val.aggregate().item()
        avg_sensitivity = sens_metric_bin_val.aggregate()[0].item()
        dice_metric_bin_val.reset()
        sens_metric_bin_val.reset()        
        
        print(f"[Epoch {epoch+1}] Val Loss: {avg_val_loss:.4f} | Dice class 1 vs rest: {avg_dice:.4f}")

        run.log({
            "epoch_val_loss": avg_val_loss,
            "epoch_val_dice_class1_vs_rest": avg_dice,
            "epoch": epoch + 1,
            "epoch_val_dice": avg_dice,
            "epoch_val_sensitivity": avg_sensitivity,
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