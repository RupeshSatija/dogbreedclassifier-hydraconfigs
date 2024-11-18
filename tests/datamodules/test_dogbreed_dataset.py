import pytest
import rootutils
from torchvision import transforms

# Setup root directory
root = rootutils.setup_root(__file__, pythonpath=True)

from src.datamodules.dogbreed_dataset import DogBreedDataModule


@pytest.fixture
def datamodule_config():
    return {
        "dir": "data/dogbreed",
        "batch_size": 32,
        "num_workers": 2,
        "pin_memory": True,
        "train_val_test_split": [0.7, 0.15, 0.15],
        "image_size": 224,
        "crop_size": 224,
    }


@pytest.fixture
def datamodule(datamodule_config):
    return DogBreedDataModule(**datamodule_config)


def test_datamodule_init(datamodule, datamodule_config):
    for key, value in datamodule_config.items():
        assert getattr(datamodule, key) == value


def test_prepare_data_directory_not_exists(datamodule, tmp_path):
    datamodule.dir = str(tmp_path / "nonexistent")
    with pytest.raises(RuntimeError, match="Data directory .* not found"):
        datamodule.prepare_data()


def test_prepare_data_directory_exists(datamodule, tmp_path):
    test_dir = tmp_path / "dogbreed"
    test_dir.mkdir()
    datamodule.dir = str(test_dir)
    datamodule.prepare_data()  # Should not raise any exception


def test_transforms_properties(datamodule):
    # Test normalize transform
    assert isinstance(datamodule.normalize_transform, transforms.Normalize)

    # Test train transform
    assert isinstance(datamodule.train_transform, transforms.Compose)
    transform_list = datamodule.train_transform.transforms
    assert any(isinstance(t, transforms.RandomResizedCrop) for t in transform_list)
    assert any(isinstance(t, transforms.RandomHorizontalFlip) for t in transform_list)
    assert any(isinstance(t, transforms.ColorJitter) for t in transform_list)
    assert any(isinstance(t, transforms.RandomRotation) for t in transform_list)

    # Test valid transform
    assert isinstance(datamodule.valid_transform, transforms.Compose)
    transform_list = datamodule.valid_transform.transforms
    assert any(isinstance(t, transforms.Resize) for t in transform_list)
    assert any(isinstance(t, transforms.CenterCrop) for t in transform_list)


def test_transform_sizes(datamodule):
    crop_transform = [
        t
        for t in datamodule.train_transform.transforms
        if isinstance(t, transforms.RandomResizedCrop)
    ][0]
    assert crop_transform.size == (datamodule.crop_size, datamodule.crop_size)

    resize_transform = [
        t
        for t in datamodule.valid_transform.transforms
        if isinstance(t, transforms.Resize)
    ][0]
    assert resize_transform.size == (datamodule.image_size, datamodule.image_size)
