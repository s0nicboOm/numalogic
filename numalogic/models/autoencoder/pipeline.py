import logging

import numpy as np
from numpy.typing import NDArray
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import TQDMProgressBar
from torch.utils.data import DataLoader

from numalogic.tools.types import AutoencoderModel

_LOGGER = logging.getLogger(__name__)

from pytorch_lightning.callbacks import Callback


class DefaultCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        print("Training is starting")

    def on_train_end(self, trainer, pl_module):
        print("Training is ending")


class AutoencoderPipeline(Trainer):
    r"""
    Class to simplify training, inference, loading and saving of time-series autoencoders.

    Note:
         This class only supports Pytorch models.
    Args:
        model: model instance
        seq_len: sequence length
        loss_fn: loss function used for training
                        supported values include    {"huber", "l1", "mse"}
        optimizer: optimizer to used for training.
                           supported values include {"adam", "adagrad", "rmsprop"}
        lr: learning rate
        batch_size: batch size for training
        num_epochs: number of epochs for training
                              can be used when the threshold calculated is too low
        resume_train: parameter to decide if resume training is needed. Also,
                              based on this parameter the optimizer state dict
                              is stored in registry.

    >>> # Example usage
    >>> from numalogic.models.autoencoder.variants import VanillaAE
    >>> x = np.random.randn(100, 3)
    >>> seq_len = 10
    >>> model = VanillaAE(signal_len=seq_len, n_features=3)
    >>> ae_trainer = AutoencoderPipeline(model=model, seq_len=seq_len)
    >>> ae_trainer.fit(x)
    """

    def __init__(
        self, model: AutoencoderModel = None, seq_len: int = None, batch_size=64, **kwargs
    ):
        super().__init__(**kwargs)
        self.callbacks = [DefaultCallback()]
        if not (model and seq_len):
            raise ValueError("No model and seq len provided!")
        self.batch_size = batch_size
        self._model = model
        self.seq_len = seq_len

    @property
    def model(self) -> AutoencoderModel:
        return self._model

    def create_loader(self, X: NDArray[float]):
        data = self._model.construct_dataset(
            X, batch=self.batch_size, seq_len=self.seq_len, returnDataLoader=True
        )
        return data

    def score(self, score_func, input: NDArray[float]):
        input_dataloader = self.create_loader(input)
        output = self.predict(model=self._model, dataloaders=input_dataloader)
        recon_err = score_func(input_dataloader.dataset.data.numpy(), output[0].numpy())
        return recon_err
