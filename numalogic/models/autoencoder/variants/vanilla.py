from typing import Tuple, Sequence, Any, Optional, Union

import torch
from torch import nn, Tensor, optim

import pytorch_lightning as pl
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchmetrics.functional import f1_score, precision, recall

from numalogic.preprocess.datasets import SequenceDataset
from numalogic.tools.exceptions import LayerSizeMismatchError


class _Encoder(nn.Module):
    r"""
    Encoder module for the autoencoder module.

    Args:
        seq_len: sequence length / window length
        n_features: num of features
        layersizes: encoder layer size
        dropout_p: the dropout value

    """

    def __init__(self, seq_len: int, n_features: int, layersizes: Sequence[int], dropout_p: float):
        super().__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.dropout_p = dropout_p

        layers = self._construct_layers(layersizes)
        self.encoder = nn.Sequential(*layers)

    def _construct_layers(self, layersizes: Sequence[int]) -> nn.ModuleList:
        r"""
        Utility function to generate a simple feedforward network layer

        Args:
            layersizes: layer size

        Returns:
            A simple feedforward network layer of type nn.ModuleList
        """
        layers = nn.ModuleList()
        start_layersize = self.seq_len

        for lsize in layersizes[:-1]:
            layers.extend(
                [
                    nn.Linear(start_layersize, lsize),
                    nn.BatchNorm1d(self.n_features),
                    nn.Tanh(),
                    nn.Dropout(p=self.dropout_p),
                ]
            )
            start_layersize = lsize

        layers.extend(
            [
                nn.Linear(start_layersize, layersizes[-1]),
                nn.BatchNorm1d(self.n_features),
                nn.LeakyReLU(),
            ]
        )
        return layers

    def forward(self, x: Tensor) -> Tensor:
        return self.encoder(x)


class _Decoder(nn.Module):
    r"""
    Decoder module for the autoencoder module.

    Args:
        seq_len: sequence length / window length
        n_features: num of features
        layersizes: decoder layer size
        dropout_p: the dropout value

    """

    def __init__(self, seq_len: int, n_features: int, layersizes: Sequence[int], dropout_p: float):
        super().__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.dropout_p = dropout_p
        layers = self._construct_layers(layersizes)
        self.decoder = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.decoder(x)

    def _construct_layers(self, layersizes: Sequence[int]) -> nn.ModuleList:
        r"""
        Utility function to generate a simple feedforward network layer

        Args:
            layersizes: layer size

        Returns:
            A simple feedforward network layer
        """
        layers = nn.ModuleList()

        for idx, _ in enumerate(layersizes[:-1]):
            layers.extend(
                [
                    nn.Linear(layersizes[idx], layersizes[idx + 1]),
                    nn.BatchNorm1d(self.n_features),
                    nn.Tanh(),
                    nn.Dropout(p=self.dropout_p),
                ]
            )

        layers.append(nn.Linear(layersizes[-1], self.seq_len))
        return layers


class VanillaAE(pl.LightningModule):
    r"""
    Vanilla Autoencoder model comprising Fully connected layers only.

    Args:

        signal_len: sequence length / window length
        n_features: num of features
        encoder_layersizes: encoder layer size (default = Sequence[int] = (16, 8))
        decoder_layersizes: decoder layer size (default = Sequence[int] = (8, 16))
        dropout_p: the dropout value (default=0.25)
        loss_fn = loss function used for model training
    """

    def __init__(
        self,
        signal_len: int,
        n_features: int = 1,
        encoder_layersizes: Sequence[int] = (16, 8),
        decoder_layersizes: Sequence[int] = (8, 16),
        dropout_p: float = 0.25,
        loss_fn=None,
    ):

        super().__init__()
        self.loss_fn = F.mse_loss
        if loss_fn:
            self.loss_fn = loss_fn
        self.seq_len = signal_len
        self.dropout_prob = dropout_p
        self.save_hyperparameters()
        if encoder_layersizes[-1] != decoder_layersizes[0]:
            raise LayerSizeMismatchError(
                f"Last layersize of encoder: {encoder_layersizes[-1]} "
                f"does not match first layersize of decoder: {decoder_layersizes[0]}"
            )

        self.encoder = _Encoder(
            seq_len=signal_len,
            n_features=n_features,
            layersizes=encoder_layersizes,
            dropout_p=dropout_p,
        )
        self.decoder = _Decoder(
            seq_len=signal_len,
            n_features=n_features,
            layersizes=decoder_layersizes,
            dropout_p=dropout_p,
        )

        self.encoder.apply(self.init_weights)
        self.decoder.apply(self.init_weights)

    @staticmethod
    def init_weights(m: nn.Module) -> None:
        r"""
        Initiate parameters in the transformer model.
        """
        if type(m) == nn.Linear:
            nn.init.xavier_normal_(m.weight)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.2, patience=10, min_lr=1e-5
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "train_loss"}

    def _get_reconstruction_loss(self, batch):
        x = batch
        x_hat = self.forward(x)
        loss = self.loss_fn(x, x_hat)
        loss = loss.mean()
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, _precision, _recall, f1 = self._shared_eval_step(batch, batch_idx)
        metrics = {
            "val_loss": loss,
            "val_precision": _precision,
            "val_recall": _recall,
            "test_f1_score": f1,
        }
        self.log_dict(metrics)
        return metrics

    def test_step(self, batch, batch_idx):
        loss, _precision, _recall, f1 = self._shared_eval_step(batch, batch_idx)
        metrics = {
            "test_loss": loss,
            "test_precision": _precision,
            "test_recall": _recall,
            "test_f1_score": f1,
        }
        self.log_dict(metrics)
        return metrics

    def _shared_eval_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        _loss = F.cross_entropy(y_hat, y)
        _precision = precision(y_hat, y, task="multiclass")
        _recall = recall(y_hat, y, task="multiclass")
        _f1 = f1_score(y_hat, y, task="multiclass")
        return _loss, _precision, _recall, _f1

    def construct_dataset(
        self, x: Tensor, batch: int = None, seq_len: int = None, returnDataLoader=False
    ) -> Union[SequenceDataset, DataLoader]:
        r"""
         Constructs dataset given tensor and seq_len

         Args:
            x: Tensor type
            seq_len: sequence length / window length
            batch: batch size

        Returns:
            SequenceDataset type
        """
        __seq_len = seq_len or self.seq_len
        dataset = SequenceDataset(x, __seq_len, permute=True)
        if returnDataLoader:
            dataset_loader = DataLoader(dataset, batch_size=batch, shuffle=False)
            return dataset_loader
        return dataset
