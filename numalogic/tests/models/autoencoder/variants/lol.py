import os
import time

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from numalogic._constants import TESTS_DIR
from numalogic.models.autoencoder import AutoencoderPipeline
from numalogic.models.threshold._std import StdDevThreshold
from numalogic.postprocess import tanh_norm
from numalogic.preprocess.datasets import SequenceDataset
from numalogic.preprocess.transformer import LogTransformer

ROOT_DIR = os.path.join(TESTS_DIR, "resources", "data")
DATA_FILE = os.path.join(ROOT_DIR, "interactionstatus.csv")

from numalogic.models.autoencoder.variants import VanillaAE

model = VanillaAE(10)
df = pd.read_csv(DATA_FILE)
df = df[["failure", "success"]]
df_t = pd.DataFrame(
    data=[
        [1, 2],
        [3, 4],
        [5, 6],
        [7, 8],
        [9, 10],
        [1001, 10002],
        [1, 2],
        [3, 4],
        [5, 6],
        [7, 8],
        [9, 10],
        [11, 12],
        [1, 2],
        [5, 6]
    ],
    columns=["tt", "lol"],
)

scaler = LogTransformer()
X_train = scaler.fit_transform(df[:-240])
X_test = scaler.fit_transform(df_t[["tt", "lol"]])
pl1 = AutoencoderPipeline(
    model=VanillaAE(12, n_features=2),
    seq_len=12,
    batch_size=512,
    accelerator="auto",
    devices=1 if torch.cuda.is_available() else None,
    max_epochs=1,
    log_every_n_steps=1,
    amp_backend='apex',
    amp_level='02'
)


def reco_loss(a, b):
    return np.square(a - b)


train_loader = pl1.create_loader(X_train)
test_loader = pl1.create_loader(X_test)
pl1.fit(model=pl1.model, train_dataloaders=train_loader)
f_score = pl1.score(score_func=reco_loss, input=X_test)
th = StdDevThreshold()
t=time.time()
th.fit(train_loader.dataset.data.numpy())
print(time.time()-t)
test_scores = th.predict(f_score).mean(axis=1)
tanh_score = tanh_norm(test_scores)
print(np.mean(tanh_score, axis=1))

traced_script_module = torch.jit.trace(pl1.model, train_loader)
traced_script_module.save("traced_train.pt")
