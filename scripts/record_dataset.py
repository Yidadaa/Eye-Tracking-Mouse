"""[summary.

Returns:
    [type]: [description.
"""
import json
import os
import ctypes
from pathlib import Path
from typing import Tuple

import cv2
from mtcnn.mtcnn import MTCNN
import pyWinhook as pyHook
from pyWinhook.HookManager import MouseEvent
import pythoncom
import mtcnn
import numpy as np

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

class Recorder(object):
  def __init__(self, output: Path, w:int = 2560, h:int = 1440, load_ck: bool = True) -> None:
    super().__init__()
    self.window_size = (w, h)
    self.load_ck = load_ck
    self.record_list = []
    self.output_path = self.check_path(output)
    self.meta_path = self.output_path / 'meta.json'
    self.img_path = self.check_path(self.output_path / 'img')
    self.load_meta() # load old files or not
    # face detector
    self.detector = MTCNN()
    self.scale_factor = 1
    # face and eye size
    self.face_size = (200, 200)
    self.eye_size = (64, 64)
    # capture and keyboard hook
    self.cap = cv2.VideoCapture(0)
    self.cap_size = (0, 0)
    self.hook = self.setup_hook()
    self.key_queue = []

  def load_meta(self):
    print('[Recorder]: loading meta')
    if self.meta_path.exists() and self.load_ck:
      meta = json.loads(self.meta_path.read_text())
      self.record_list = meta['record']
      print(f'[Record] load meta file with {len(self.record_list)} images')

  def build_item(self, frame: np.ndarray, frame_index:int, write_disk:bool = True) -> Tuple[dict, dict]:
    """Build record according to frame.

    Args:
        frame (np.ndarray): capture frame
        frame_index (int): frame index
        write_disk (bool, optional): write to disk or not. Defaults to True.

    Returns:
        Tuple[dict, dict]: faces meta data, faces image data
    """
    h, w = frame.shape[:2]
    self.cap_size = (w, h)
    sw, sh = w // self.scale_factor, h // self.scale_factor
    # use smaller image to detect faces
    smaller_frame = cv2.cvtColor(cv2.resize(frame, (sw, sh)), cv2.COLOR_BGR2RGB)
    faces = self.detector.detect_faces(smaller_frame)
    faces_meta = []
    faces_data = [] # return face images
    for face_index, face in enumerate(faces):
      face_data = {
        'face': None,
        'eyes': []
      }
      face_meta = {
        'box': None,
        'face_center': None,
        'eyes': [],
        'face_img': None
      }
      eyes = self.get_eye_region(face['keypoints'])
      x, y, w, h = [x * self.scale_factor for x in face['box']]
      w, h = max(w, h), max(w, h) # use long side as region
      # save face image
      face_image = cv2.resize(frame[y:y + h, x:x + w], self.face_size)
      face_meta['box'] = [x, y, w, h]
      face_meta['face_center'] = [x + w // 2, y + h // 2]
      face_path = self.img_path / f'{frame_index}_face_{face_index}.jpg'
      face_meta['face_img'] = str(face_path.relative_to(self.output_path))
      face_data['face'] = face_image
      if write_disk:
        cv2.imwrite(str(face_path), face_image)
      # save eye image
      for eye_side, eye_pos in zip(['left', 'right'], eyes):
        eye_pos = [[int(k) for k in self.scale_factor * pos] for pos in eye_pos]
        (sx, sy), (ex, ey) = eye_pos
        eye_image = cv2.resize(frame[sy:ey, sx:ex], self.eye_size)
        eye_path = self.img_path / f'{frame_index}_face_{face_index}_{eye_side}_eye.jpg'
        if write_disk:
          cv2.imwrite(str(eye_path), eye_image)
        face_meta['eyes'].append([eye_pos, str(eye_path.relative_to(self.output_path))])
        face_data['eyes'].append(eye_image)
      faces_data.append(face_data)
      faces_meta.append(face_meta)
    return faces_meta, faces_data, faces

  def save_record(self, event: MouseEvent):
    '''Record data. '''
    frame_index = len(self.record_list)
    frame = self.capture()
    faces_meta, _, __ = self.build_item(frame, frame_index)

    # dont save image with no faces
    if len(faces_meta) == 0:
      return

    # save origin images
    fname = self.img_path / f'{frame_index}.jpg'
    cv2.imwrite(str(fname), frame)
    self.record_list.append([
      event.Position,
      str(fname.relative_to(self.output_path)),
      faces_meta
    ])
    print(f'[Record] Rec_{len(self.record_list)}: {event.Position} with {fname}')

  def write_meta(self):
    meta = {
      'count': len(self.record_list),
      'record': self.record_list,
      'window_size': self.window_size,
      'eye_size': self.eye_size,
      'face_size': self.face_size,
      'cap_size': self.cap_size
    }
    with open(str(self.meta_path), 'w') as f:
      json.dump(meta, f)
    print(f'[Record] meta file written to: {self.meta_path}')

  def check_path(self, p: Path) -> Path:
    if not self.load_ck:
      for f in p.iterdir():
        if f.is_file(): f.unlink()
      print(f'[Recorder]: old {p} removed')
    if not p.exists():
      p.mkdir()
      print(f'[Recorder]: {p} created')
    return p

  def get_eye_region(self, kpts: dict) -> list:
    '''Get eye region from mtcnn result. '''
    le_pos, re_pos = np.array(kpts['left_eye']), np.array(kpts['right_eye'], dtype=int)
    center_x, center_y = (le_pos + re_pos) // 2
    eyes = []
    for pos in [le_pos, re_pos]:
      half_w = np.abs(center_x - pos[0])
      left_top = pos - half_w
      right_bottom = pos + half_w
      eyes.append([left_top, right_bottom])
    return eyes

  def setup_hook(self):
    hm = pyHook.HookManager()
    # register two callbacks
    hm.MouseAllButtonsDown = self.on_mouse_event
    hm.KeyDown = self.on_keyboard_event

    # hook into the mouse and keyboard events
    hm.HookMouse()
    hm.HookKeyboard()
    return hm

  def capture(self) -> np.ndarray:
    ret, frame = self.cap.read()
    return frame

  def run(self):
    """Run recorded."""
    pythoncom.PumpMessages()

  def exit(self):
    self.write_meta()
    ctypes.windll.user32.PostQuitMessage(0)

  def on_mouse_event(self, event: pyHook.MouseEvent):
    print(f'[Record] mouse down: {event.MessageName}')
    self.save_record(event)
    return True

  def on_keyboard_event(self, event: pyHook.KeyboardEvent):
    print(f'[Record] key pressed: {event.Key}')
    self.key_queue.append(event.Key)
    if len(self.key_queue) > 4:
      self.key_queue.pop(0)
    if ''.join(self.key_queue) == 'QUIT':
      self.exit()
    return True

if __name__ == '__main__':
  output = Path(__file__).parent / 'output'
  rec = Recorder(output, load_ck=True)
  rec.run()
