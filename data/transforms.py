import monai
from torch import float32

def get_deterministic_transforms(cfg):
    return monai.transforms.Compose([
      monai.transforms.LoadImaged(keys=['pet', 'ct', 'totseg', 'mask'], image_only=False),
      monai.transforms.EnsureChannelFirstd(keys=['pet', 'ct', 'totseg', 'mask'], channel_dim="no_channel"),
      monai.transforms.Orientationd(keys=["pet", "ct", 'totseg', "mask"], axcodes='RAS'),
      monai.transforms.Spacingd(keys=['pet', 'mask'], pixdim=(3.0, 3.0, 3.0), mode=('bilinear', 'nearest')),
      monai.transforms.ResampleToMatchd(keys=['ct', 'totseg'], key_dst='pet', mode=('bilinear', 'nearest')),
      monai.transforms.CastToTyped(keys=['pet', 'ct'], dtype=float32),
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