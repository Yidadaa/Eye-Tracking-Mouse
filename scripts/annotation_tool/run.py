'''This file is used for making annotations.'''

import os
from pathlib import Path

port = 8080

os.chdir(Path(__file__).parent.parent)
print(f'Annotation page at: http://localhost:{port}/annotation_tool')
os.system('python -m http.server 8080')