import argparse
import os
from pathlib import Path

from ultralytics.utils.downloads import download

def visdrone2yolo(dir):
  from PIL import Image
  from tqdm import tqdm

  def convert_box(size, box):
      # Convert VisDrone box to YOLO xywh box
      dw = 1. / size[0]
      dh = 1. / size[1]
      return (box[0] + box[2] / 2) * dw, (box[1] + box[3] / 2) * dh, box[2] * dw, box[3] * dh

  (dir / 'labels').mkdir(parents=True, exist_ok=True)  # make labels directory
  pbar = tqdm((dir / 'annotations').glob('*.txt'), desc=f'Converting {dir}')
  for f in pbar:
      img_size = Image.open((dir / 'images' / f.name).with_suffix('.jpg')).size
      lines = []
      with open(f, 'r') as file:  # read annotation.txt
          for row in [x.split(',') for x in file.read().strip().splitlines()]:
              if row[4] == '0':  # VisDrone 'ignored regions' class 0
                  continue
              cls = int(row[5]) - 1
              box = convert_box(img_size, tuple(map(int, row[:4])))
              lines.append(f"{cls} {' '.join(f'{x:.6f}' for x in box)}\n")
              with open(str(f).replace(f'{os.sep}annotations{os.sep}', f'{os.sep}labels{os.sep}'), 'w') as fl:
                  fl.writelines(lines)  # write label.txt



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root-dir', default='./data', type=str,
                        help="root path of images and labels")

    arg = parser.parse_args()
    # Convert
    for d in 'VisDrone2019-DET-train', 'VisDrone2019-DET-val', 'VisDrone2019-DET-test-dev':
        visdrone2yolo(arg.root_dir / d)  # convert VisDrone annotations to YOLO labels
