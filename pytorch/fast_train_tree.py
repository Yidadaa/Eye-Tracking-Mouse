from pathlib import Path
import json

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import numpy as np
from tqdm import tqdm

# classifier output size
classifier_size = 5

# load data
meta_path = Path(Path(__file__).parent / '../scripts/output/meta_faces.json')
meta = json.loads(meta_path.read_text())

screen_size = np.array(meta['window_size'])

train_data = []
train_positions = []
for record in tqdm(meta['record'], desc='[Train] Loading'):
  pos, _, __, face_feature = record
  train_data.append(face_feature)
  pos = (np.array(pos) / screen_size * classifier_size).astype(np.int)
  flatten_pos = pos[0] + pos[1] * classifier_size
  pos_index = int(flatten_pos)
  train_positions.append(pos_index)
train_positions = np.stack(train_positions)

X_train, X_test, y_train, y_test = train_test_split(
  train_data, train_positions, test_size=0.2)

print(f'[Train] label shape: {y_train.shape} {y_train[:1]}')

# clf = KNeighborsClassifier()
clf = lgb.LGBMClassifier(n_estimators=100)
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))