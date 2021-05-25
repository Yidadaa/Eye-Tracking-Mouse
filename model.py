import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18

class EyeTrackerModel(nn.Module):
  def __init__(self, w:int = 640, h:int = 480):
    super().__init__()
    self.encoder = self.get_encoder()

  def get_encoder(self):
    resnet = resnet18(pretrained=True)
    resnet.fc = nn.Linear(in_features=512, out_features=2)
    return resnet

  def forward(self, x:torch.Tensor) -> torch.Tensor:
    return self.encoder(x)

if __name__ == '__main__':
  w, h = 640, 480
  model = EyeTrackerModel(w, h)
  dummy = torch.rand((1, 3, w, h))
  encoder_output = model.encoder(dummy)
  predict_output = model(dummy)
  print('Encoder shape: ', encoder_output.shape)
  print('Model output shape: ', predict_output.shape)