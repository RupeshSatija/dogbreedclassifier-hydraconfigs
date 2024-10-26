from unittest.mock import MagicMock, patch

import lightning as L
import pytest

from src.datamodules.dogbreed_dataset import DogBreedDataModule
from src.eval import evaluate, load_model_from_checkpoint, main
from src.models.dogbreed_classifier import DogBreedGenericClassifier


@pytest.fixture
def mock_data_module():
    return MagicMock(spec=DogBreedDataModule)


@pytest.fixture
def mock_model():
    return MagicMock(spec=DogBreedGenericClassifier)


@pytest.fixture
def mock_trainer():
    trainer = MagicMock(spec=L.Trainer)
    trainer.validate.return_value = [{"val_loss": 0.3, "val_accuracy": 0.85}]
    return trainer


def test_load_model_from_checkpoint(mock_model):
    with patch.object(
        DogBreedGenericClassifier, "load_from_checkpoint", return_value=mock_model
    ):
        model = load_model_from_checkpoint("dummy_path.ckpt")
        assert isinstance(model, DogBreedGenericClassifier)


def test_evaluate(mock_data_module, mock_model, mock_trainer):
    with patch("src.eval.L.Trainer", return_value=mock_trainer):
        evaluate(mock_data_module, mock_model)
        mock_trainer.validate.assert_called_once_with(
            mock_model, datamodule=mock_data_module
        )


@patch("src.eval.setup_logger")
@patch("src.eval.DogBreedDataModule")
@patch("src.eval.load_model_from_checkpoint")
@patch("src.eval.evaluate")
def test_main(mock_evaluate, mock_load_model, mock_data_module, mock_setup_logger):
    mock_args = MagicMock()
    mock_args.checkpoint = "dummy_checkpoint.ckpt"

    with patch("src.eval.argparse.ArgumentParser.parse_args", return_value=mock_args):
        main()

    mock_setup_logger.assert_called_once()
    mock_data_module.assert_called_once()
    mock_load_model.assert_called_once_with("dummy_checkpoint.ckpt")
    mock_evaluate.assert_called_once()


@pytest.mark.parametrize(
    "val_results",
    [
        [{"val_loss": 0.3, "val_accuracy": 0.85}],
        [{"val_loss": 0.5, "val_accuracy": 0.75}],
    ],
)
def test_evaluate_different_results(mock_data_module, mock_model, val_results):
    mock_trainer = MagicMock(spec=L.Trainer)
    mock_trainer.validate.return_value = val_results

    with patch("src.eval.L.Trainer", return_value=mock_trainer):
        evaluate(mock_data_module, mock_model)
        mock_trainer.validate.assert_called_once_with(
            mock_model, datamodule=mock_data_module
        )


def test_main_file_not_found():
    with pytest.raises(FileNotFoundError):
        with patch("src.eval.argparse.ArgumentParser.parse_args") as mock_args:
            mock_args.return_value.checkpoint = "non_existent_file.ckpt"
            main()
