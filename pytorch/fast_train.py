from pathlib import Path

import torch
from torch.optim import Adam

from skorch import NeuralNetRegressor
from sklearn.model_selection import train_test_split

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import pickle, json

from model import EyeTrackerModel
from dataset import EyeDataset

eye_model = EyeTrackerModel()

meta_path = Path(Path(__file__).parent / '../scripts/output/meta.json')
eye_ds = EyeDataset(meta_path)
train_data = []
train_positions = []
for i in tqdm(range(len(eye_ds)), desc='[Train] Loading'):
  face, eyes, grid, pos = eye_ds[i]
  train_data.append((face, eyes, grid))
  train_positions.append(pos)
train_positions = torch.stack(train_positions)

X_train, X_test, y_train, y_test = train_test_split(train_data, train_positions)

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

print(f'[Train] label shape: {y_train.shape}')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
net = NeuralNetRegressor(EyeTrackerModel, max_epochs=300, optimizer=Adam,
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