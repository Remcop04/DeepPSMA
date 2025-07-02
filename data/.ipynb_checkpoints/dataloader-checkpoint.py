import os
from monai.data import PersistentDataset, DataLoader, Dataset
from data.transforms import get_deterministic_transforms, get_random_transforms

def build_dict_petct(data_path, tracer):
    pt_folders = [f for f in os.listdir(data_path)]
    paths_scan = [os.path.join(data_path, path, tracer) for path in pt_folders]
    dicts = []

    for scan_path in paths_scan:
        mask_path = os.path.join(scan_path, "TTB.nii.gz")
        ct_path = os.path.join(scan_path, "CT.nii.gz")
        pet_path = os.path.join(scan_path, "PET.nii.gz")
        totseg_path = os.path.join(scan_path, "totseg_24.nii.gz")
        if os.path.exists(ct_path):
            dicts.append({'pet': pet_path, 'ct': ct_path, 'mask': mask_path, 'totseg': totseg_path})
    return dicts

def get_dataloader(cfg):
    train_dict = build_dict_petct(cfg["data_path"], cfg["tracer"])
    
    print("Create persistent dataset")
    persistent_ds = PersistentDataset(train_dict, transform=get_deterministic_transforms(cfg), cache_dir=cfg["cache_dir"])
    train_dataset = Dataset(data=persistent_ds, transform=get_random_transforms(cfg))

    print("Return dataloader")
    return DataLoader(train_dataset, batch_size=cfg["batch_size"], shuffle=True, num_workers=2)
