from typing import List
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import mobilenetv3

class EyeTrackerModel(nn.Module):
  """Eye Tracker Model."""

  def __init__(self, face_size:int = 200, eye_size:int = 64, grid_size:int = 5,
      face_dim:int = 128, eye_dim:int = 64, grid_dim:int = 64, out_dim:int = 25):
    """Eye tracking model.

    Args:
        face_size (int, optional): face image size. Defaults to 200.
        eye_size (int, optional): eye image size. Defaults to 64.
        grid_size (int, optional): face position grid. Defaults to 25.
        face_dim (int, optional): face feature dims. Defaults to 128.
        eye_dim (int, optional): eye feature dims. Defaults to 64.
        grid_dim (int, optional): position feature dims. Defaults to 64.
    """
    super().__init__()
    self.face_size = face_size
    self.eye_size = eye_size
    self.grid_size = grid_size
    self.face_dim = face_dim
    self.eye_dim = eye_dim
    self.grid_dim = grid_dim
    self.grid_encoder = nn.Sequential(
      nn.Flatten(),
      nn.Linear(grid_size * grid_size, 256),
      nn.ReLU(),
      nn.Linear(256, grid_dim)
    )
    self.face_encoder = self.get_encoder(out_dim=face_dim)
    self.eye_encoder = self.get_encoder(out_dim=eye_dim)
    self.decoder = nn.Sequential(
      nn.Linear(face_dim + eye_dim * 2 + grid_dim, 128),
      nn.ReLU(),
      nn.Linear(128, out_dim),
      nn.Softmax(1)
    )

  def get_encoder(self, out_dim:int = 64) -> nn.Module:
    """Get backbone encoder.

    Args:
        out_dim (int, optional): encoder's output dim. Defaults to 64.

    Returns:
        nn.Module: a pretrained model
    """
    encoder = mobilenetv3.mobilenet_v3_large(pretrained=True)
    encoder.classifier[-1] = nn.Linear(in_features=1280, out_features=out_dim)
    return encoder

  def forward(self, face:torch.Tensor, eyes:torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """Forward method.

    Args:
        face (torch.Tensor): face image tensor
        eyes (torch.Tensor): eye image tensor with (n, 2, 3, w, h)
        grid (torch.Tensor): position grid tensor

    Returns:
        torch.Tensor: predicted position (x, y), scaled to [1, 1]
    """
    face_feature = self.face_encoder(face)
    eye_feature: torch.Tensor = self.eye_encoder(eyes.view((-1, 3, self.eye_size, self.eye_size)))
    eye_feature = eye_feature.view((-1, self.eye_size * 2))
    grid_feature = self.grid_encoder(grid)
    feature = torch.cat((face_feature, eye_feature, grid_feature), dim=1)
    y_pred = self.decoder(feature)
    return y_pred

if __name__ == '__main__':
  model = EyeTrackerModel()
  inputs = []
  for shape, sub_model, name in [
    ((1, 3, 64, 64), model.eye_encoder, 'eye'),
    ((1, 3, 200, 200), model.face_encoder, 'face'),
    ((1, 1, 5, 5), model.grid_encoder, 'grid'),
  ]:
    dummy = torch.rand(shape)
    inputs.append(dummy)
    output = sub_model(dummy)
    print(f'{name} shape: ', output.shape)
  inputs[1] = torch.rand((1, 2, 3, 64, 64))
  output = model(*inputs)
  print('total output: ', output.shape)
