from torch.utils.data.dataset import Dataset
from torchvision.transforms.functional import InterpolationMode
from torchvision import transforms
import numpy as np

class EyeDataset(Dataset):
  def __init__(self, path: str) -> None:
    super().__init__()
    self.transform = transforms.Compose([
      transforms.ToPILImage(),
      transforms.Resize((64, 64), interpolation=InterpolationMode.NEAREST),
      transforms.ToTensor()
    ])
    self.raw_data = self.build_data(np.load(path))

  def build_data(self, x) -> torch.Tensor:
    return [self.transform(img) for img in x]

  def __len__(self):
    return len(self.raw_data)

  def __getitem__(self, index) -> torch.Tensor:
    return self.raw_data[index]