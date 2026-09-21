"""Cached preprocessing and per-access augmentation."""

from torch.utils.data import Dataset
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureTyped,
    ScaleIntensityd,
    Resized,
    RandFlipd,
    MapTransform,
    ConcatItemsd,
    SelectItemsd,
)


class ExtractImagePaths(MapTransform):
    """Expose nested modality paths for MONAI dictionary transforms."""

    def __init__(self, keys):
        super().__init__(keys)

    def __call__(self, data):
        for k in self.keys:
            if k in data["images"]:
                data[k] = data["images"][k]
        return data


class AugmentedDataset(Dataset):
    """Apply per-access augmentation after loading persistently cached data."""

    def __init__(self, cached_dataset, transform=None):
        self.cached_dataset = cached_dataset
        self.transform = transform

    def __len__(self):
        return len(self.cached_dataset)

    def __getitem__(self, i):
        data = self.cached_dataset[i]

        if self.transform:
            data = self.transform(data)
        return data


def build_pre_transforms(img_keys, spatial_size=(120, 120, 64)):
    """Build deterministic loading, resizing, scaling, and channel concatenation."""
    return Compose(
        [
            ExtractImagePaths(keys=img_keys),
            LoadImaged(
                keys=img_keys, ensure_channel_first=True, reader="NibabelReader", image_only=True
            ),
            EnsureTyped(keys=img_keys),
            Resized(keys=img_keys, spatial_size=spatial_size, mode="trilinear"),
            ScaleIntensityd(keys=img_keys, minv=0.0, maxv=1.0),
            ConcatItemsd(keys=img_keys, name="image", dim=0),
            EnsureTyped(keys=["image"]),
        ]
    )


def build_aug_transforms(train=True):
    """Apply training-only flips and retain the four model input fields."""
    aug = []
    if train:
        aug.extend(
            [
                RandFlipd(keys=["image"], spatial_axis=[0], prob=0.5),
                RandFlipd(keys=["image"], spatial_axis=[1], prob=0.5),
            ]
        )

    aug.append(SelectItemsd(keys=["image", "text_emb", "event", "time"]))
    return Compose(aug)
