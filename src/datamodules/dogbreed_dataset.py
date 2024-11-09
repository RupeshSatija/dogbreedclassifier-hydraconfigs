import os
from typing import Optional

import lightning as L
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder


class DogBreedDataModule(L.LightningDataModule):
    def __init__(
        self,
        dir: str,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        train_val_test_split: list,
        image_size: int,
        crop_size: int,
    ):
        super().__init__()
        self.dir = dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.train_val_test_split = train_val_test_split
        self.image_size = image_size
        self.crop_size = crop_size

        self.train_dataset = self.val_dataset = self.test_dataset = None
        self.class_names = None

    def prepare_data(self):
        # DVC will handle the data pulling, so we just verify the directory exists
        if not os.path.exists(self.dir):
            raise RuntimeError(
                f"Data directory {self.dir} not found. Please run 'dvc pull' to fetch the data."
            )

    @property
    def normalize_transform(self):
        return transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    @property
    def train_transform(self):
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(self.crop_size),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                ),
                transforms.RandomRotation(20),
                transforms.ToTensor(),
                self.normalize_transform,
            ]
        )

    @property
    def valid_transform(self):
        return transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.CenterCrop(self.crop_size),
                transforms.ToTensor(),
                self.normalize_transform,
            ]
        )

    def setup(self, stage: Optional[str] = None):
        self.prepare_data()

        full_dataset = ImageFolder(self.dir, transform=self.train_transform)
        self.class_names = full_dataset.classes

        total_size = len(full_dataset)
        train_size = int(self.train_val_test_split[0] * total_size)
        val_size = int(self.train_val_test_split[1] * total_size)
        test_size = total_size - train_size - val_size

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            full_dataset, [train_size, val_size, test_size]
        )

        self.val_dataset.dataset.transform = self.valid_transform
        self.test_dataset.dataset.transform = self.valid_transform

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def get_class_names(self):
        return self.class_names
