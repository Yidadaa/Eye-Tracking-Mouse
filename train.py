from pytorch_lightning import callbacks
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pathlib import Path

from model import EyeTrackerModel
from dataset import EyeDataset

class EyeTrackingMouse(pl.LightningModule):
  def __init__(self, w:int = 256, h:int = 256):
    super(EyeTrackingMouse, self).__init__()
    self.model = EyeTrackerModel(w, h)

  def forward(self, x):
    return self.model(x)

  def configure_optimizers(self):
    return torch.optim.Adam(self.parameters(), lr=1e-2)

  def build_loss(self, batch) -> torch.Tensor:
    X, y = batch
    y_pred = self.model(X)
    loss = F.mse_loss(y, y_pred)
    return loss

  def training_step(self, batch, bidx) -> torch.Tensor:
    loss = self.build_loss(batch)
    self.log('train_loss', loss)
    return loss

  def validation_step(self, batch, bidx) -> torch.Tensor:
    loss = self.build_loss(batch)
    self.log('val_loss', loss)
    return loss

if __name__ == '__main__':
  eye_dataset = EyeDataset(Path(__file__).parent / './scripts/output/meta.json')
  test_size = int(0.2 * len(eye_dataset))
  train_size = len(eye_dataset) - test_size
  eye_train_loader, eye_test_loader = [DataLoader(ds, batch_size=64) for ds in random_split(eye_dataset, [train_size, test_size])]
  model = EyeTrackingMouse()
  checkpoint_callback = ModelCheckpoint(monitor='val_loss')
  trainer = pl.Trainer(callbacks=[checkpoint_callback])
  trainer.fit(model, eye_train_loader, eye_test_loader)