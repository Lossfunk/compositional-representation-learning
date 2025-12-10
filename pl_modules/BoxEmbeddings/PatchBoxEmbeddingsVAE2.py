from torch import nn
import lightning as L

from models import VanillaVAE


class PatchBoxEmbeddingsVAE2(L.LightningModule):
    def __init__(self, config):
        super(PatchBoxEmbeddingsVAE2, self).__init__()
        self.config = config

    def forward(self, x):
        pass

    def training_step(self, batch, batch_idx):
        pass
