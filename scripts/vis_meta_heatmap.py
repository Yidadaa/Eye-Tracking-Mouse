from pathlib import Path
import json

from matplotlib import pyplot as plt
import numpy as np


def run(meta_path: Path):
  meta_file = json.loads(meta_path.read_text())
  scale_factor = 32
  w, h = 2560, 1440
  canvas = np.zeros((h // scale_factor, w // scale_factor))
  for record in meta_file['record']:
    pos = record[0]
    x, y = pos[0] // scale_factor, pos[1] // scale_factor
    canvas[y, x] += 1
  print('[Draw] record {} points'.format(len(meta_file['record'])))
  plt.imshow(canvas)
  plt.show()


if __name__ == '__main__':
  meta_path = Path(__file__).parent / 'output/meta.json'
  run(meta_path)