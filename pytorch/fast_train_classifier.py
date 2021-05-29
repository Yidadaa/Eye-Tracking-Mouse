from pathlib import Path

import torch
from torch._C import dtype
from torch.optim import Adam

from skorch import NeuralNetClassifier
from sklearn.model_selection import train_test_split

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import pickle, json

from model import EyeTrackerModel
from dataset import EyeDataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'

screen_w, screen_h = 2560, 1440
classifier_size = 5 # train a classifier
eye_model = EyeTrackerModel(out_dim=classifier_size * classifier_size)

meta_path = Path(Path(__file__).parent / '../scripts/output/meta.json')
eye_ds = EyeDataset(meta_path)
train_data = []
train_positions = []
for i in tqdm(range(len(eye_ds)), desc='[Train] Loading'):
  face, eyes, grid, pos = eye_ds[i]
  train_data.append((face, eyes, grid))
  pos = (pos * classifier_size).long()
  flatten_pos = pos[0] + pos[1] * classifier_size
  pos_index = int(flatten_pos)
  label = torch.zeros((classifier_size * classifier_size, )).long()
  label[pos_index] = 1
  train_positions.append(torch.Tensor([pos_index]).long())
train_positions = torch.stack(train_positions).reshape(len(train_positions))

X_train, X_test, y_train, y_test = train_test_split(
  train_data, train_positions, test_size=0.05)

Xs = []
for name, data in zip(('train', 'test'), [X_train, X_test]):
  ds_dict = {
    'face': [],
    'eyes': [],
    'grid': []
  }
  for face, eyes, grid in data:
    ds_dict['face'].append(face)
    ds_dict['eyes'].append(eyes)
    ds_dict['grid'].append(grid)
  for key in ds_dict:
    ds_dict[key] = torch.stack(ds_dict[key])
    print(f'[Train] {name} dataset {key} shape {ds_dict[key].shape}')
  Xs.append(ds_dict)
X_train, X_test = Xs

print(f'[Train] label shape: {y_train.shape} {y_train[:1]}')

net = NeuralNetClassifier(EyeTrackerModel, max_epochs=100, optimizer=Adam,
    optimizer__lr=.001, device=device, batch_size=32)

net.fit(X_train, y_train)

# result = net.predict_proba(X_test)
# result = result * 255
# result = result.transpose(0, 2, 3, 1)
# result = result.astype(np.uint8)

output_path = Path(__file__).parent / 'log'
if not output_path.exists():
  output_path.mkdir()

# for i, img in tqdm(enumerate(result)):
#   Image.fromarray(img).save(output_path / '{}_pred.jpg'.format(i))
#   Image.fromarray(train_images[i]).save(output_path / '{}_orig.jpg'.format(i))

history = net.history
train_losses = history[:, 'train_loss']
val_losses = history[:, 'valid_loss']

plt.figure()
plt.plot(train_losses)
plt.plot(val_losses)
plt.savefig(output_path / 'history.jpg')

# save model
with open(output_path / 'model.pkl', 'wb') as f:
  pickle.dump(net, f)