import cv2
from PIL import Image
import pickle
from pathlib import Path
# import torch
# import torchvision.transforms as T
import numpy as np

# from skorch import NeuralNetRegressor

import win32api, win32con
import os

from mtcnn import MTCNN

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

class Model():
  def __init__(self) -> None:
    print('Loading model')
    model_path = Path(__file__).parent / './log/model.pkl'
    model: NeuralNetRegressor = pickle.loads(model_path.read_bytes())

    dummy = torch.rand((1, 3, 160, 120)).to(model.device)
    output = model.predict_proba(dummy)
    print('Test output: ', output)
    self.model = model

  def predict(self, img: np.ndarray) -> np.ndarray:
    x = Image.fromarray(img).resize((160, 120))
    x = T.ToTensor()(x).to(self.model.device).unsqueeze(0)
    pos = self.model.predict_proba(x)
    return pos

cap = cv2.VideoCapture(0)
i = 0
W, H = 2560, 1440
sw, sh = 160, 120

detector = MTCNN()
q = []

while True:
  key = cv2.waitKey(1)
  ret, frame = cap.read()
  frame = cv2.flip(frame, 1)
  smaller_frame = cv2.resize(frame, (sw, sh))
  smaller_frame = cv2.cvtColor(smaller_frame, cv2.COLOR_BGR2RGB)
  faces = detector.detect_faces(smaller_frame)
  for face in faces:
    (x0, y0, w, h) = face['box']
    cv2.rectangle(frame, (x0 * 4, y0 * 4), ((x0 + w) * 4, (y0 + h) * 4), color=(255, 0, 0), thickness=2)
    x, y = x0 + w / 2, y0 + h / 2
    x, y = x / sw, y / sh
    x, y = np.clip([x - 0.2, y - 0.2], 0, 1)
    x, y = x * W, y * H
    q.append([x, y])
    if len(q) > 5:
      q.pop(0)
    x, y = np.mean(q, 0)
    print(x, y)
    win32api.SetCursorPos((int(x), int(y)))
  if key == 27: break
  cv2.imshow('cap', frame)

cap.release()
cv2.destroyAllWindows()