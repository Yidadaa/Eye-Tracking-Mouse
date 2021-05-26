"""[summary.

Returns:
    [type]: [description.
"""
import json
import os
import ctypes
from pathlib import Path

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
    self.output_path = self.check_path(output)
    self.meta_path = self.output_path / 'meta.json'
    self.img_path = self.check_path(self.output_path / 'img')
    self.load_meta() # load old files or not
    # face detector
    self.detector = MTCNN()
    self.scale_factor = 2
    # face and eye size
    self.face_size = (200, 200)
    self.eye_size = (64, 64)
    # capture and keyboard hook
    self.cap = cv2.VideoCapture(0)
    self.hook = self.setup_hook()
    self.key_queue = []
    self.record_list = []

  def load_meta(self):
    if self.meta_path.exists() and self.load_ck:
      meta = json.loads(self.meta_path.read_text())
      self.record_list = meta['record']
      print(f'[Record] load meta file with {len(self.record_list)} images')

  def save_record(self, event: MouseEvent):
    '''Record data. '''
    frame_index = len(self.record_list)
    frame = self.capture()
    print(frame.shape)
    h, w = frame.shape[:2]
    sw, sh = w // self.scale_factor, h // self.scale_factor
    # use smaller image to detect faces
    smaller_frame = cv2.cvtColor(cv2.resize(frame, (sw, sh)), cv2.COLOR_BGR2RGB)
    faces = self.detector.detect_faces(smaller_frame)
    for face_index, face in enumerate(faces):
      eyes = self.get_eye_region(face['keypoints'])
      x, y, w, h = [x * self.scale_factor for x in face['box']]
      w, h = max(w, h), max(w, h) # use long side as region
      # save face image
      face_image = cv2.resize(frame[y:y + h, x:x + w], self.face_size)
      face_path = self.img_path / f'{frame_index}_face_{face_index}.jpg'
      face['face_img'] = str(face_path.relative_to(self.output_path))
      cv2.imwrite(str(face_path), face_image)
      # save eye image
      face['eyes'] = []
      for eye_side, eye_pos in zip(['left', 'right'], eyes):
        eye_pos = [list(self.scale_factor * pos) for pos in eye_pos]
        (sx, sy), (ex, ey) = eye_pos
        eye_image = cv2.resize(frame[sy:ey, sx:ex], self.eye_size)
        eye_path = self.img_path / f'{frame_index}_face_{face_index}_{eye_side}_eye.jpg'
        cv2.imwrite(str(eye_path), eye_image)
        face['eyes'].append([eye_pos, eye_path.relative_to(self.output_path)])

    # save origin images
    fname = self.img_path / f'{frame_index}.jpg'
    cv2.imwrite(str(fname), frame)
    self.record_list.append([
      event.Position,
      str(fname.relative_to(self.output_path))
    ])
    print(f'[Record] Rec_{len(self.record_list)}: {event.Position} with {fname}')

  def write_meta(self):
    meta = {
      'count': len(self.record_list),
      'record': self.record_list,
      'window_size': self.window_size,
      'eye_size': self.eye_size,
      'face_size': self.face_size
    }
    with open(str(self.meta_path), 'w') as f:
      json.dump(meta, f)
    print(f'[Record] meta file written to: {self.meta_path}')

  def check_path(self, p: Path) -> Path:
    if not self.load_ck:
      for f in p.iterdir():
        if f.is_file(): f.unlink()
    if not p.exists():
      p.mkdir()
      print(f'[Recorder]: {p} created')
    return p

  def get_eye_region(self, kpts: dict) -> list:
    '''Get eye region from mtcnn result. '''
    le_pos, re_pos = np.array(kpts['left_eye']), np.array(kpts['right_eye'])
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
    if len(self.key_queue) > 1:
      self.key_queue.pop(0)
    if ''.join(self.key_queue) == 'Escape':
      self.exit()
    return True

if __name__ == '__main__':
  output = Path(__file__).parent / 'output'
  rec = Recorder(output, load_ck=False)
  rec.run()
