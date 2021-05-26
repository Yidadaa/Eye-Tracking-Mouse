from pathlib import Path

import torch
from torch.optim import Adam

from skorch import NeuralNetRegressor
from sklearn.model_selection import train_test_split

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from tqdm import tqdm
import pickle, json

from model import EyeTrackerModel

eye_model = EyeTrackerModel()

meta_path = Path(Path(__file__).parent / './scripts/output/meta.json')
meta = json.loads(meta_path.read_text())
train_images = []
train_positions = []
for pos, img_path in tqdm(meta['record'], desc='Loading'):
  train_images.append(np.array(Image.open(meta_path.parent / img_path).resize((160, 120))))
  train_positions.append(list(pos))

train_images: np.ndarray = np.stack(train_images)
train_positions = np.array(train_positions, dtype=np.float32)
train_data = train_images.transpose(0, 3, 1, 2)
train_data = (train_data / 255).astype(np.float32)

X_train, X_test, y_train, y_test = train_test_split(train_data, train_positions)

print(X_train.shape, y_train.shape)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
net = NeuralNetRegressor(EyeTrackerModel, max_epochs=300, optimizer=Adam,
    optimizer__lr=.01, device=device, batch_size=64)

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