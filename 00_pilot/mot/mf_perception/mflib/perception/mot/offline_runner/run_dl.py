import numpy as np
import os
from ultralytics import YOLO

import cv2



images_dir = '/workspace/data/seedling/images'

dl_type = 'pose' # det, seg, pose

# weights = '/workspace/Metafarmers/ai_models/seedling/yolo11m-det.pt'
weights = f'/workspace/Metafarmers/ai_models/seedling/yolo11m-{dl_type}.pt'
# weights = '/workspace/Metafarmers/ai_models/seedling/yolo11m-seg.pt'

output_dir = f'/workspace/data/seedling/{dl_type}_out'

def main():
  # prepare images
  image_names = os.listdir(images_dir)
  image_paths = [os.path.join(images_dir, image_name) for image_name in image_names if image_name.endswith('.jpg') or image_name.endswith('.png')]

  image_paths.sort(key=lambda x: int(x.split('/')[-1].split('.')[0]))

  # prepare model
  model = YOLO(weights)

  # run detection
  for idx, i_name in enumerate(image_paths):
    img = cv2.imread(i_name)

    results = model(img)
    results[0].save(os.path.join(output_dir, f'{idx:05d}.jpg'))



if __name__ == '__main__':
  main()
  # bag read bag