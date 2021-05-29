'''Update faces info to meta file.'''

from pathlib import Path
import json
import os

import numpy as np
import cv2
from mtcnn import MTCNN
from numpy.core.records import record
from tqdm import tqdm

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


def run():
  detector = MTCNN()
  meta_file = Path(__file__).parent / './output/meta.json'
  output_meta_file = Path(__file__).parent / './output/meta_faces.json'
  meta = json.loads(meta_file.read_text())
  frame_size = np.array(meta['cap_size'])
  valid_face = 0
  for i in tqdm(range(len(meta['record'])), '[Update] detecting'):
    record = meta['record'][i]
    face_path = meta_file.parent / record[1]
    face_img = cv2.imread(str(face_path))
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(face_img)
    if len(faces) == 0:
      continue
    valid_face += 1
    features = []
    face_start = np.array(faces[0]['box'][:2])
    face_size = np.array(faces[0]['box'][2:])
    for k in faces[0]['keypoints']:
      features.append((faces[0]['keypoints'][k] - face_start) / face_size)
    features.append((face_start + face_size / 2) / frame_size) # add face position
    features: np.ndarray = np.stack(features).reshape(len(features) * 2)
    record.append(features.tolist())
  output_meta_file.write_text(json.dumps(meta))
  print(f'[Update] Done {valid_face} / {len(meta["record"])}')

if __name__ == '__main__':
  run()