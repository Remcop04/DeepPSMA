from monai.transforms import MapTransform
from monai.data.meta_tensor import MetaTensor
from torch import float32

import torch
import numpy as np
import monai
import json

class ConvertToMultiClassLabel(MapTransform):
    def __init__(self, mask_key, pet_key, threshold_key):
        super().__init__(keys=[mask_key], allow_missing_keys=True)
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

class SelectTracerKeysPostCache(MapTransform):
    def __init__(self, tracer="PSMA"):
        self.tracer = tracer
        self.pet_key = f"{tracer}_pet"
        self.mask_key = f"{tracer}_mask"
        self.ct_key = f"{tracer}_ct"
        self.totseg_key = f"{tracer}_totseg"
        self.pet_copy_key = f"{tracer}_pet_copy"

    def __call__(self, data):
        return {
            self.pet_key: data[self.pet_key],
            self.mask_key: data[self.mask_key],
            self.ct_key: data[self.ct_key],
            self.totseg_key: data[self.totseg_key],
            self.pet_copy_key: data[self.pet_copy_key],
        }

class ApplySUVthreshold(MapTransform):
    def __init__(self, pet_key, threshold_key):
        super().__init__(keys=[pet_key], allow_missing_keys=True)
        self.pet_key = pet_key
        self.threshold_key = threshold_key

    def __call__(self, data):
        d = dict(data)
        
        with open(d[self.threshold_key],'r') as f:
            threshold = json.load(f)['suv_threshold']
        pet_array = d[self.pet_key]
        d[self.pet_key][pet_array < threshold] = 0

        return d

class MaskOrgansOutd(MapTransform):
    def __init__(self, pet_key: str, seg_key: str, allow_missing_keys: bool = False):
        super().__init__(keys=[pet_key, seg_key], allow_missing_keys=True)
        self.pet_key = pet_key
        self.seg_key = seg_key
        
        if "PSMA" in pet_key:
            self.organs_to_mask = [1, 2, 3, 4, 18, 19, 20, 21]
        else:
            self.organs_to_mask = [1, 2, 3, 4, 18, 19, 20, 21, 90] # Also remove brain
    
    def __call__(self, data):
        d = dict(data)
        pet = d[self.pet_key]
        seg = d[self.seg_key]

        for organ in self.organs_to_mask:
            mask = seg == organ
            pet[mask] = 0

            
        pet[pet > 0] = 1
        d[self.pet_key] = pet
        
        return d      

def get_deterministic_transforms():
   return monai.transforms.Compose([
     monai.transforms.LoadImaged(keys=['pet', 'ct', 'totseg', 'mask'], image_only=False),
     monai.transforms.EnsureChannelFirstd(keys=['pet', 'ct', 'totseg', 'mask'], channel_dim="no_channel"),
     monai.transforms.Orientationd(keys=["pet", "ct", 'totseg', "mask"], axcodes='RAS'),
     ConvertToMultiClassLabel('mask', 'pet', 'threshold'),
     monai.transforms.Spacingd(keys=['pet', 'mask'], pixdim=(3.0, 3.0, 3.0), mode=('bilinear', 'nearest')),
     monai.transforms.ResampleToMatchd(keys=['ct', 'totseg'], key_dst='pet', mode=('bilinear', 'nearest')),
     monai.transforms.CastToTyped(keys=['pet', 'ct'], dtype=torch.float32),
     monai.transforms.CopyItemsd(keys=['pet'], times=1, names=['pet_copy']),
     ApplySUVthreshold('pet_copy', 'threshold'),
     #monai.transforms.Flipd(keys=["pet", "ct", 'totseg', "mask"], spatial_axis=0),
     monai.transforms.ScaleIntensityRanged(keys=['pet'], a_min=0, a_max=50, b_min=0.0, b_max=1.0, clip=True),
     monai.transforms.ScaleIntensityRanged(keys=['ct'], a_min=-200, a_max=1000, b_min=0.0, b_max=1.0, clip=True),
     MaskOrgansOutd(pet_key='pet_copy', seg_key='totseg'),
     monai.transforms.EnsureTyped(keys=['pet', 'ct', 'totseg', 'mask'])
])

def get_random_transforms():
   return monai.transforms.Compose([
     monai.transforms.RandAffined(keys=['pet', 'pet_copy', 'ct', 'totseg', 'mask'], prob=0.5, rotate_range=(0.1, 0.1, 0.1), scale_range=(0.1, 0.1, 0.1),
     translate_range=(10, 10, 10), padding_mode='border', mode=('bilinear', 'nearest', 'bilinear', 'nearest', 'nearest')),
     monai.transforms.RandGaussianNoised(keys=["pet", "ct"], prob=0.3, mean=0.0, std=0.05),
     monai.transforms.RandShiftIntensityd(keys=["pet"], offsets=0.1, prob=0.5),
     monai.transforms.RandScaleIntensityd(keys=["pet"], factors=0.1, prob=0.5),
     monai.transforms.RandBiasFieldd(keys=["ct"], prob=0.3),
     #monai.transforms.RandRotated(keys=['pet', 'ct', 'totseg', 'mask'], range_x=0.34906585, prob=0.2, mode=['bilinear','bilinear', 'nearest', 'nearest']), # 20 degrees
     monai.transforms.SpatialPadd(keys=["pet", "ct", 'totseg', "mask"], spatial_size=(128, 128, 128)),
     #monai.transforms.RandSpatialCropd(keys=['pet', 'ct', 'totseg', 'mask'], roi_size=(128, 128, 128), random_center=True, random_size=False),
     monai.transforms.RandCropByPosNegLabeld(keys=["ct", "pet", 'pet_copy', 'totseg', "mask"], label_key="mask", spatial_size=(128, 128, 128), pos=0.8, neg=0.2, num_samples=4, image_key="pet")
])

# from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd, ResampleToMatchd, CastToTyped, CopyItemsd, ScaleIntensityRanged, EnsureTyped, RandAffined, RandGaussianNoised, RandShiftIntensityd, RandScaleIntensityd, RandBiasFieldd, SpatialPadd, RandCropByPosNegLabeld


# def get_transforms(cfg):
#   # ---------- Configuratie ----------
#   base_keys = ['pet', 'ct', 'totseg', 'mask']
#   prefixes = ['PSMA_', 'FDG_']
  
#   # ---------- Hulpfunctie ----------
#   def prefixed_keys(keys, prefix):
#       return [prefix + k for k in keys]
  
#   # ---------- Bouwen van transform pipeline ----------
#   transforms = []
#   random_transforms = []
  
#   # Loop over PSMA_ en FDG_ blokken
#   for prefix in prefixes:
#       keys = prefixed_keys(base_keys, prefix)
  
#       # Unpack voor leesbaarheid
#       threshold_key = prefix + 'threshold'
#       pet_key = prefix + 'pet'
#       ct_key = prefix + 'ct'
#       totseg_key = prefix + 'totseg'
#       mask_key = prefix + 'mask'
#       pet_copy_key = prefix + 'pet_copy'
  
#       # Laad en orienteer beelden
#       transforms += [
#           LoadImaged(keys=keys, image_only=False, allow_missing_keys=True),
#           EnsureChannelFirstd(keys=keys, channel_dim="no_channel"),
#           Orientationd(keys=keys, axcodes='RAS'),
#           ConvertToMultiClassLabel(mask_key, pet_key, threshold_key),    
#           Spacingd(keys=[pet_key, mask_key], pixdim=(3.0, 3.0, 3.0), mode=('bilinear', 'nearest')),
#           ResampleToMatchd(keys=[ct_key, totseg_key], key_dst=pet_key, mode=('bilinear', 'nearest')),
#           CastToTyped(keys=[pet_key, ct_key], dtype=torch.float32),
#           CopyItemsd(keys=[pet_key], times=1, names=[pet_copy_key]),
#           ApplySUVthreshold(pet_copy_key, threshold_key),
#           ScaleIntensityRanged(keys=[pet_key, pet_copy_key], a_min=0, a_max=50, b_min=0.0, b_max=1.0, clip=True),
#           ScaleIntensityRanged(keys=[ct_key], a_min=-200, a_max=1000, b_min=0.0, b_max=1.0, clip=True),
#           MaskOrgansOutd(pet_key=pet_copy_key, seg_key=totseg_key),
#           EnsureTyped(keys=[pet_key, pet_copy_key, ct_key, totseg_key, mask_key]),]
  
#       # Augmentaties
#       random_transforms += [
#           RandAffined(keys=[pet_key, pet_copy_key, ct_key, totseg_key, mask_key], prob=0.5, rotate_range=(0.1, 0.1, 0.1), scale_range=(0.1, 0.1, 0.1),
#                       translate_range=(10, 10, 10), padding_mode='border', mode=('bilinear', 'bilinear', 'bilinear', 'nearest', 'nearest')),
#           RandGaussianNoised(keys=[pet_key, ct_key], prob=0.3, mean=0.0, std=0.05),
#           RandShiftIntensityd(keys=[pet_key], offsets=0.1, prob=0.5),
#           RandScaleIntensityd(keys=[pet_key], factors=0.1, prob=0.5),
#           RandBiasFieldd(keys=[ct_key],prob=0.3),
#           #SpatialPadd(keys=[pet_key, ct_key, totseg_key, mask_key],spatial_size=(128, 128, 128)),
#           RandCropByPosNegLabeld(keys=[ct_key, pet_key, pet_copy_key, totseg_key, mask_key], label_key=mask_key, image_key=pet_key, spatial_size=(128, 128, 128), pos=0.8,neg=0.2, num_samples=4)]
      
      
#   # ---------- Combineer alles ----------
#   deterministic_transform = Compose(transforms)
#   random_transform = Compose(random_transforms)
#   return deterministic_transform, random_transform
