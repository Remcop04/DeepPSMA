import os
import json

from monai.data import PersistentDataset, Dataset, DataLoader
from sklearn.model_selection import train_test_split
from data.transforms import get_deterministic_transforms, get_random_transforms

def build_dict_petct(data_path):
    """
    Returns two lists of dictionaries (PSMA and FDG), each containing keys 'img_pet', 'img_ct', 'mask', and 'patient_id'.

    Args:
        data_path (str): path to the root folder of the data set.

    Returns:
        Tuple[List[Dict], List[Dict]]: list of PSMA dicts and list of FDG dicts
    """

    psma_dicts = []
    fdg_dicts = []

    pt_folders = [f for f in os.listdir(data_path) if not f.startswith(".")]
    
    for patient_id in pt_folders:
        base_path = os.path.join(data_path, patient_id)

        for tracer, out_list in [('PSMA', psma_dicts), ('FDG', fdg_dicts)]:
            scan_path = os.path.join(base_path, tracer)
            pet_path = os.path.join(scan_path, "PET.nii.gz")
            ct_path = os.path.join(scan_path, "CT.nii.gz")
            mask_path = os.path.join(scan_path, "TTB.nii.gz")
            totseg_path = os.path.join(scan_path, "totseg_24.nii.gz")
            threshold_path = os.path.join(scan_path, "threshold.json")

            if not (os.path.exists(pet_path) and os.path.exists(ct_path) and os.path.exists(mask_path)):
                continue

            sample_dict = {
                "patient_id": patient_id,
                "pet": pet_path,
                "ct": ct_path,
                "mask": mask_path,
                "totseg": totseg_path if os.path.exists(totseg_path) else None,
                "threshold": threshold_path if os.path.exists(threshold_path) else None,
            }

            out_list.append(sample_dict)

    return psma_dicts, fdg_dicts

def split_by_patient_id(dicts, val_split=0.2, seed=42):
    patient_ids = [d["patient_id"] for d in dicts]
    train_ids, val_ids = train_test_split(patient_ids, test_size=val_split, random_state=seed)

    train = [d for d in dicts if d["patient_id"] in train_ids]
    val = [d for d in dicts if d["patient_id"] in val_ids]

    return train, val
    
def get_dataloaders(cfg):
    
    psma_data, fdg_data = build_dict_petct(cfg["data_path"])
    
    tracer = cfg['tracer']
    if tracer == "PSMA":
        train_files, val_files = split_by_patient_id(psma_data, val_split=0.2)
    else:
        train_files, val_files = split_by_patient_id(fdg_data, val_split=0.2)
        
    #train_files, val_files = train_test_split(data, test_size=cfg.get("val_split", 0.2), random_state=cfg.get("seed", 42))

    print(f"Train: {len(train_files)}, Val: {len(val_files)}")

    train_ds = PersistentDataset(train_files, transform=get_deterministic_transforms(), cache_dir=cfg[f"{tracer}_cache_dir"])
    train_ds = Dataset(data=train_ds, transform=get_random_transforms())

    val_ds = PersistentDataset(val_files, transform=get_deterministic_transforms(), cache_dir=cfg[f"{tracer}_cache_dir"])

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg.get("num_workers", 4))
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=cfg.get("num_workers", 4))

    return train_loader, val_loader
