from config import cfg
from data.dataloader import get_dataloaders
from models.unet import get_unet
from train.trainer import train
from utils.wandb_utils import setup_wandb

import torch

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Setup wandb")
    run = setup_wandb(cfg)
    
    print("Start dataloader")
    train_loader, val_loader = get_dataloaders(cfg)
    model = get_unet(cfg).to(device)

    print("Start training loop")
    train(cfg, model, train_loader, val_loader, device, run)

if __name__ == "__main__":
    main()
