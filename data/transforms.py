from monai.transforms import MapTransform
from torch import float32

import torch
import numpy as np
import monai
import json

class ConvertToMultiClassLabel(MapTransform):
    def __init__(self, keys, mask_key='mask', pet_key='pet', threshold_key='threshold'):
        super().__init__(keys)
        self.mask_key = mask_key
        self.pet_key = pet_key
        self.threshold_key = threshold_key

    def __call__(self, data):
        d = dict(data)
        mask_array = d[self.mask_key]
        pet_array = d[self.pet_key]
        with open(d[self.threshold_key],'r') as f:
            threshold = json.load(f)['suv_threshold']

        multiclass_label = np.zeros(mask_array.shape, dtype=np.uint8)
        multiclass_label[mask_array > 0] = 1  # metastase
        multiclass_label[np.logical_and(pet_array >= threshold, mask_array == 0)] = 2  # fysiologisch

        # Maak nieuwe MetaTensor aan, kopieer metadata van origineel mask
        if isinstance(mask_array, MetaTensor):
            new_label = MetaTensor(multiclass_label, affine=mask_array.affine)
        else:
            new_label = multiclass_label  # fallback, kan ook np.ndarray zijn

        d[self.mask_key] = new_label

        return d

def get_deterministic_transforms(cfg):
    return monai.transforms.Compose([
      monai.transforms.LoadImaged(keys=['pet', 'ct', 'totseg', 'mask'], image_only=False),
      monai.transforms.EnsureChannelFirstd(keys=['pet', 'ct', 'totseg', 'mask'], channel_dim="no_channel"),
      monai.transforms.Orientationd(keys=["pet", "ct", 'totseg', "mask"], axcodes='RAS'),
      ConvertToMultiClassLabel(keys=['mask', 'pet', 'threshold']),
      monai.transforms.Spacingd(keys=['pet', 'mask'], pixdim=(3.0, 3.0, 3.0), mode=('bilinear', 'nearest')),
      monai.transforms.ResampleToMatchd(keys=['ct', 'totseg'], key_dst='pet', mode=('bilinear', 'nearest')),
      monai.transforms.CastToTyped(keys=['pet', 'ct'], dtype=torch.float32),
      #monai.transforms.Flipd(keys=["pet", "ct", 'totseg', "mask"], spatial_axis=0),
      monai.transforms.ScaleIntensityRanged(keys=['pet'], a_min=0, a_max=50, b_min=0.0, b_max=1.0, clip=True),
      monai.transforms.ScaleIntensityRanged(keys=['ct'], a_min=-200, a_max=1000, b_min=0.0, b_max=1.0, clip=True),
      monai.transforms.EnsureTyped(keys=['pet', 'ct', 'totseg', 'mask'])
])

def get_random_transforms(cfg):
    return monai.transforms.Compose([
      monai.transforms.RandAffined(keys=['pet', 'ct', 'totseg', 'mask'], prob=0.5, rotate_range=(0.1, 0.1, 0.1), scale_range=(0.1, 0.1, 0.1),
      translate_range=(10, 10, 10), padding_mode='border', mode=('bilinear', 'bilinear', 'nearest', 'nearest')),
      monai.transforms.RandGaussianNoised(keys=["pet", "ct"], prob=0.3, mean=0.0, std=0.05),
      monai.transforms.RandShiftIntensityd(keys=["pet"], offsets=0.1, prob=0.5),
      monai.transforms.RandScaleIntensityd(keys=["pet"], factors=0.1, prob=0.5),
      monai.transforms.RandBiasFieldd(keys=["ct"], prob=0.3),
      #monai.transforms.RandRotated(keys=['pet', 'ct', 'totseg', 'mask'], range_x=0.34906585, prob=0.2, mode=['bilinear','bilinear', 'nearest', 'nearest']), # 20 degrees
      monai.transforms.SpatialPadd(keys=["pet", "ct", 'totseg', "mask"], spatial_size=(128, 128, 128)),
      #monai.transforms.RandSpatialCropd(keys=['pet', 'ct', 'totseg', 'mask'], roi_size=(128, 128, 128), random_center=True, random_size=False),
      monai.transforms.RandCropByPosNegLabeld(keys=["ct", "pet", 'totseg', "mask"], label_key="mask", spatial_size=(128, 128, 128), pos=0.8, neg=0.2, num_samples=4, image_key="pet")
])
