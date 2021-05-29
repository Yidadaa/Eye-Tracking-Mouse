from pathlib import Path
import json
from typing import List, Tuple

import torch
from torch.utils.data.dataset import Dataset
from torchvision.transforms.functional import InterpolationMode
from torchvision import transforms
import numpy as np
from PIL import Image
from tqdm import tqdm

class EyeDataset(Dataset):
  def __init__(self, meta_path: Path, grid_size:int = 5) -> None:
    super().__init__()
    self.grid_size = grid_size
    self.meta_path = meta_path
    self.transform_ori = transforms.Compose([
      transforms.Resize((160, 120), interpolation=InterpolationMode.NEAREST),
      transforms.ToTensor()
    ])
    self.transform_to_tensor = transforms.ToTensor()
    self.raw_data = self.build_data(meta_path)

  def build_data(self, meta_path: Path) -> list:
    """Load data from meta file.

    Args:
        meta_path (Path): meta file path

    Returns:
        list: raw data list
    """
    assert meta_path.exists(), f'[Dataset] meta file not exists: {meta_path}'
    self.meta = json.loads(meta_path.read_text())
    ret_list = []
    for record in tqdm(self.meta['record'], desc='[Dataset] building dataset'):
      pos, rel_img_path, faces = record
      img_path, face_path, eye_paths = self.get_face_and_eyes_img_path(rel_img_path)
      if face_path.exists():
        ret_list.append([pos, img_path, face_path, eye_paths, faces[0]['box']])
    return ret_list

  def __len__(self):
    return len(self.raw_data)

  def get_face_and_eyes_img_path(self, rel_img_path:str) -> Tuple[Path, Path, List[Path]]:
    img_path = self.meta_path.parent / rel_img_path
    img_index = img_path.stem
    face_path = img_path.parent / f'{img_index}_face_0.jpg'
    eye_paths = [img_path.parent / f'{img_index}_face_0_{side}_eye.jpg'
      for side in ['left', 'right']]
    return img_path, face_path, eye_paths

  def __getitem__(self, index) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor, torch.Tensor]:
    """Get dataset item.

    Args:
      index (int): item index

    Returns:
      torch.Tensor: face image tensor
      List[torch.Tensor]: eye images tensor
      torch.Tensor: grid tensor with (grid_size, grid_size)
      torch.Tensor: pos with (x, y)
    """
    assert 'cap_size' in self.meta, '[Dataset] please check if cap_size in meta json file'
    pos, img_path, face_path, eye_paths, face_box = self.raw_data[index]
    # scale posotion to [0, 1]
    pos = list(x / s for x, s in zip(pos, self.meta['window_size']))
    face_image = self.transform_to_tensor(Image.open(face_path))
    eyes_image = torch.stack([self.transform_to_tensor(Image.open(f)) for f in eye_paths])
    pos_gt = torch.Tensor(pos)
    face_box = torch.Tensor([(x / z, y / z) for x, y, z in zip(face_box[:2], face_box[2:], self.meta['cap_size'])])
    (x, w), (y, h) = (face_box * self.grid_size).int()
    w, h = max(1, w), max(1, h) # ensure w >= 1 and h >= 1
    grid = torch.zeros((self.grid_size, self.grid_size))
    grid[y:y + h, x:x + w] = 1
    return face_image, eyes_image, grid, pos_gt


if __name__ == '__main__':
  ds = EyeDataset(Path(__file__).parent / '../scripts/output/meta.json')
  test_item = ds[0]
  for item in test_item:
    print(item.shape if type(item) is not list else item[0].shape)
