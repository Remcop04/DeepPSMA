import monai
from torch import float32

# def get_train_transforms(cfg):
#     return Compose([
#         LoadImaged(keys=['pet', 'ct', 'totseg', 'mask'], image_only=False),
#         CastToTyped(keys=['pet', 'ct'], dtype=torch.float32),
#         EnsureChannelFirstd(keys=['pet', 'ct', 'totseg', 'mask']),
#         Orientationd(keys=["pet", "ct", 'totseg', "mask"], axcodes='RAS'),
#         Spacingd(keys=['pet', 'ct', 'totseg', 'mask'], pixdim=cfg["spacing"] + [2.0], mode=('bilinear', 'bilinear', 'nearest', 'nearest')),
#         ResampleToMatchd(keys=['ct', 'totseg'], key_dst='pet', mode=('bilinear', 'nearest')),
#         ScaleIntensityRanged(keys=['pet'], a_min=cfg["pet_range"][0], a_max=cfg["pet_range"][1], b_min=0.0, b_max=1.0, clip=True),
#         ScaleIntensityRanged(keys=['ct'], a_min=cfg["ct_range"][0], a_max=cfg["ct_range"][1], b_min=0.0, b_max=1.0, clip=True),
#         RandRotated(keys=['pet', 'ct', 'totseg', 'mask'], range_x=0.349, prob=0.2, mode=['bilinear', 'bilinear', 'nearest', 'nearest']),
#         SpatialPadd(keys=["pet", "ct", 'totseg', "mask"], spatial_size=cfg["image_size"]),
#         RandSpatialCropd(keys=['pet', 'ct', 'totseg', 'mask'], roi_size=cfg["image_size"], random_center=True, random_size=False),
#         EnsureTyped(keys=['pet', 'ct', 'totseg', 'mask'])
#     ])

def get_deterministic_transforms(cfg):
    return monai.transforms.Compose([
      monai.transforms.LoadImaged(keys=['pet', 'ct', 'totseg', 'mask'], image_only=False),
      monai.transforms.EnsureChannelFirstd(keys=['pet', 'ct', 'totseg', 'mask'], channel_dim="no_channel"),
      monai.transforms.Orientationd(keys=["pet", "ct", 'totseg', "mask"], axcodes='RAS'),
      monai.transforms.Spacingd(keys=['pet', 'mask'], pixdim=(2.0, 2.0, 2.0), mode=('bilinear', 'nearest')),
      monai.transforms.ResampleToMatchd(keys=['ct', 'totseg'], key_dst='pet', mode=('bilinear', 'nearest')),
      monai.transforms.CastToTyped(keys=['pet', 'ct'], dtype=float32),
      #monai.transforms.Flipd(keys=["pet", "ct", 'totseg', "mask"], spatial_axis=0),
      monai.transforms.ScaleIntensityRanged(keys=['pet'], a_min=0, a_max=50, b_min=0.0, b_max=1.0, clip=True),
      monai.transforms.ScaleIntensityRanged(keys=['ct'], a_min=-200, a_max=1000, b_min=0.0, b_max=1.0, clip=True)
])

def get_random_transforms(cfg):
    return monai.transforms.Compose([
      monai.transforms.RandRotated(keys=['pet', 'ct', 'totseg', 'mask'], range_x=0.34906585, prob=0.2, mode=['bilinear','bilinear', 'nearest', 'nearest']), # 20 degrees
      monai.transforms.SpatialPadd(keys=["pet", "ct", 'totseg', "mask"], spatial_size=(128, 128, 128)),
      monai.transforms.RandSpatialCropd(keys=['pet', 'ct', 'totseg', 'mask'], roi_size=(128, 128, 128), random_center=True, random_size=False),
      monai.transforms.EnsureTyped(keys=['pet', 'ct', 'totseg', 'mask'])
])