cfg = {
    "data_path": "/scratch/p321870/deepPSMAchallenge/dataset/",
    "cache_dir": "/tmp/cache_dir",
    "tracer": "PSMA",
    "batch_size": 4,
    "epochs": 300,
    "lr": 1e-4,
    "image_size": (128, 128, 128),
    "spacing": [2.0, 2.0, 2.0],
    "pet_range": [0, 50],
    "ct_range": [-200, 1000],
    "num_workers": 2,
    "wandb": {
        "project": "monai-petct-segmentation",
        "name": "loss (dice 0.8, CE 0.2) include_background=False",
        "key": "637c41502e1a526172e998d0147681f4b3290999"
    }
}
