import wandb

def setup_wandb(cfg):
    wandb.login(key=cfg["wandb"]["key"])
    return wandb.init(
        project=cfg["wandb"]["project"],
        name=cfg["wandb"]["name"],
        config=cfg
    )
