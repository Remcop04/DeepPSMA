import os
from monai.data import PersistentDataset, Dataset, DataLoader
from sklearn.model_selection import train_test_split
from data.transforms import get_deterministic_transforms, get_random_transforms

def build_dict_petct(data_path, tracer):
    pt_folders = sorted([f for f in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, f))])
    dicts = []

    for folder in pt_folders:
        scan_path = os.path.join(data_path, folder, tracer)
        mask_path = os.path.join(scan_path, "TTB.nii.gz")
        ct_path = os.path.join(scan_path, "CT.nii.gz")
        pet_path = os.path.join(scan_path, "PET.nii.gz")
        totseg_path = os.path.join(scan_path, "totseg_24.nii.gz")
        if os.path.exists(ct_path) and os.path.exists(pet_path) and os.path.exists(mask_path):
            dicts.append({'pet': pet_path, 'ct': ct_path, 'mask': mask_path, 'totseg': totseg_path})
    return dicts

def get_dataloaders(cfg):
    data = build_dict_petct(cfg["data_path"], cfg["tracer"])
    train_files, val_files = train_test_split(data, test_size=cfg.get("val_split", 0.2), random_state=cfg.get("seed", 42))

    print(f"Total: {len(data)}, Train: {len(train_files)}, Val: {len(val_files)}")

    train_ds = PersistentDataset(train_files, transform=get_deterministic_transforms(cfg), cache_dir=cfg["cache_dir"])
    train_ds = Dataset(data=train_ds, transform=get_random_transforms(cfg))

    val_ds = PersistentDataset(val_files, transform=get_deterministic_transforms(cfg), cache_dir=cfg["cache_dir"])

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg.get("num_workers", 4))
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=cfg.get("num_workers", 4))

    return train_loader, val_loader
