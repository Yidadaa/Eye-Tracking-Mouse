"""[summary.

Returns:
    [type]: [description.
"""

import json
import ctypes
from pathlib import Path

import cv2
import pyWinhook as pyHook
from pyWinhook.HookManager import MouseEvent
import pythoncom


class Recorder(object):
  def __init__(self, output: Path, w:int = 2560, h:int = 1440) -> None:
    super().__init__()
    self.window_size = (w, h)
    self.output_path = self.check_path(output)
    self.cap = cv2.VideoCapture(0)
    self.hook = self.setup_hook()
    self.key_queue = []
    self.img_path = self.check_path(self.output_path / 'img')
    self.record_list = []

  def save_record(self, event: MouseEvent):
    frame = self.capture()
    fname = self.img_path / f'{len(self.record_list)}.jpg'
    cv2.imwrite(str(fname), frame)
    self.record_list.append([event.Position, str(fname.relative_to(self.output_path))])
    print(f'[Record] Rec_{len(self.record_list)}: {event.Position} with {fname}')

  def write_meta(self):
    meta = {
      'count': len(self.record_list),
      'record': self.record_list,
      'window_size': self.window_size
    }
    meta_path = str(self.output_path / 'meta.json')
    with open(meta_path, 'w') as f:
      json.dump(meta, f)
    print(f'[Record] meta file written to: {meta_path}')

  def check_path(self, p: Path) -> Path:
    if not p.exists():
      p.mkdir()
      print(f'[Recorder]: {p} created')
    return p

  def setup_hook(self):
    hm = pyHook.HookManager()
    # register two callbacks
    hm.MouseAllButtonsDown = self.on_mouse_event
    hm.KeyDown = self.on_keyboard_event

    # hook into the mouse and keyboard events
    hm.HookMouse()
    hm.HookKeyboard()
    return hm

  def capture(self):
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
  rec = Recorder(output)
  rec.run()
