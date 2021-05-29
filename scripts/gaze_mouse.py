import cv2
from PIL import Image
import pickle
from pathlib import Path
import numpy as np

from skorch import NeuralNetRegressor
import torch
from torchvision import transforms
from torchvision.transforms.functional import to_tensor

import win32api, win32con
import os

from mtcnn import MTCNN

from record_dataset import Recorder

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


cap = cv2.VideoCapture(0)
i = 0
W, H = 2560, 1440
cam_w, cam_h = 640, 480
sw, sh = 320, 240
scale =cam_w // sw

model = pickle.loads((Path(__file__).parent / '../pytorch/log/model.pkl').read_bytes())
recorder = Recorder(Path(__file__).parent / '.output')

q = []
last_pos = np.array([0, 0])

def draw_eye_region(frame: np.ndarray, kpts: dict):
  le_pos, re_pos = np.array(kpts['left_eye']), np.array(kpts['right_eye'])
  center_x, center_y = (le_pos + re_pos) / 2
  # draw eyes
  for pos in [le_pos, re_pos]:
    half_w = np.abs(center_x - pos[0])
    left_top = pos - half_w
    right_bottom = pos + half_w
    cv2.rectangle(frame, (left_top * scale).astype(np.int), (right_bottom * scale).astype(np.int), color=(255, 0, 0), thickness=2)
  cv2.circle(frame, (int(center_x * scale), int(center_y * scale)), 2, (0, 255, 0))
  return frame

transform = transforms.Compose([
  transforms.ToPILImage(),
  transforms.ToTensor()
])

def get_gaze_pos(faces_meta: dict, faces_data: dict):
  if len(faces_meta) == 0: return 0, 0
  face = faces_data[0]['face']
  eyes = faces_data[0]['eyes']
  face_box = faces_meta[0]['box']
  face_image: torch.Tensor = transform(face)
  eyes_image: torch.Tensor = torch.stack([transform(eye) for eye in eyes])
  face_box = torch.Tensor([(x / z, y / z) for x, y, z in zip(face_box[:2], face_box[2:], [cam_w, cam_h])])
  (x, w), (y, h) = (face_box * 5).int()
  w, h = max(1, w), max(1, h) # ensure w >= 1 and h >= 1
  grid = torch.zeros((5, 5))
  grid[y:y + h, x:x + w] = 1
  X = {
    'face': face_image.unsqueeze(0),
    'eyes': eyes_image.unsqueeze(0),
    'grid': grid.unsqueeze(0)
  }
  y_pred = model.predict_proba(X)
  return y_pred[0].reshape((5, 5))

while True:
  i += 1
  key = cv2.waitKey(1)
  ret, frame = cap.read()
  frame = cv2.flip(frame, 1)
  smaller_frame = cv2.resize(frame, (sw, sh))
  smaller_frame = cv2.cvtColor(smaller_frame, cv2.COLOR_BGR2RGB)
  faces_meta, faces_data, faces = recorder.build_item(frame, 0, write_disk=False)
  gaze = get_gaze_pos(faces_meta, faces_data)
  # gaze *= np.array([W, H])
  # gaze = gaze.astype(np.int)
  # gaze_x, gaze_y = gaze
  # print(gaze)
  gaze = cv2.resize(gaze, (320, 180))
  for face in faces:
    (x0, y0, w, h) = face['box']
    nose_x, nose_y = face['keypoints']['nose']
    cv2.rectangle(frame, (x0 * scale, y0 * scale), ((x0 + w) * scale, (y0 + h) * scale), color=(255, 0, 0), thickness=2)
    # Draw key points
    for k, v in face['keypoints'].items():
      x, y = v
      cv2.circle(frame, (x * scale, y * scale), 2, (0, 0, 255))
    # draw center and eye region
    frame = draw_eye_region(frame, face['keypoints'])
    face_x, face_y = x0 + w / 2, y0 + h / 2 # use face
    x, y = np.mean(list(face['keypoints'].values()) + [[face_x, face_y]], 0)
    x, y = x / sw, y / sh
    x, y = np.clip([x - 0.2, y - 0.2], 0, 1) * 0.2 + 0.5
    x, y = x * W, y * H
    q.append([x, y])
    if len(q) > 5:
      q.pop(0)
    pos = np.mean(q, 0)
    x, y = pos
    # x, y = gaze * 2 - 1000
    if i % 5 == 0: print(x, y)
    dxy = pos - last_pos
    # if np.sum(dxy ** 2) ** 0.5 > 0:
    #   win32api.SetCursorPos((int(x), int(y)))
    last_pos = pos
    break
  if key == 27: break
  # cv2.imshow('cap', frame)
  cv2.imshow('gaze', gaze)

cap.release()
cv2.destroyAllWindows()