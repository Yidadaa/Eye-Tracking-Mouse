from pathlib import Path
import json
from typing import Tuple

import torch
from torch.utils.data.dataset import Dataset
from torchvision.transforms.functional import InterpolationMode
from torchvision import transforms
import numpy as np
from PIL import Image

class EyeDataset(Dataset):
  def __init__(self, meta_path: Path) -> None:
    super().__init__()
    self.meta_path = meta_path
    self.transform = transforms.Compose([
      transforms.Resize((160, 120), interpolation=InterpolationMode.NEAREST),
      transforms.ToTensor()
    ])
    self.raw_data = self.build_data(meta_path)

  def build_data(self, meta_path: Path) -> list:
    self.meta = json.loads(meta_path.read_text())
    return self.meta['record']

  def __len__(self):
    return len(self.raw_data)

  def __getitem__(self, index) -> Tuple[torch.Tensor]:
    pos, rel_path = self.raw_data[index]
    pos = list(x / s for x, s in zip(pos, self.meta['window_size']))
    img_path = self.meta_path.parent / rel_path
    return self.transform(Image.open(img_path)), torch.Tensor(pos)