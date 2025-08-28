import json
import os
from datetime import datetime
import cv2
import numpy as np
from pathlib import Path

class COCOConverter:
  """Convert YOLO keypoint detection results to COCO format"""
  
  def __init__(self, categories=None):
    """
    Initialize COCO converter
    
    Args:
      categories (list): List of category dictionaries with keypoint information
    """
    self.coco_data = {
      "info": {
        "year": str(datetime.now().year),
        "version": "1.0",
        "description": "YOLO Keypoint Detection Results - SPAGRI Strawberry Dataset",
        "contributor": "SPAGRI",
        "url": "",
        "date_created": datetime.now().isoformat()
      },
      "licenses": [{"id": 1, "url": "", "name": "Unknown"}],
      "images": [],
      "annotations": [],
      "categories": categories or self._get_default_categories()
    }
    self.image_id = 1
    self.annotation_id = 1
  
  def _get_default_categories(self):
    """Get default category configuration for strawberry keypoints based on actual model output"""
    return [
      {
        "id": 1,
        "name": "partripe",
        "supercategory": "strawberry",
        "keypoints": ["fruit", "calyx", "stem"],
        "skeleton": [[1, 2], [2, 3]]
      },
      {
        "id": 2,
        "name": "ripe",
        "supercategory": "strawberry",
        "keypoints": ["fruit", "calyx", "stem"],
        "skeleton": [[1, 2], [2, 3]]
      },
      {
        "id": 3,
        "name": "unripe",
        "supercategory": "strawberry",
        "keypoints": ["fruit", "calyx", "stem"],
        "skeleton": [[1, 2], [2, 3]]
      }
    ]
  
  def add_image(self, image_path, width=None, height=None):
    """
    Add image information to COCO data
    
    Args:
      image_path (str): Path to the image file
      width (int): Image width (will be read from file if not provided)
      height (int): Image height (will be read from file if not provided)
    
    Returns:
      int: Image ID
    """
    if width is None or height is None:
      img = cv2.imread(str(image_path))
      if img is not None:
        height, width = img.shape[:2]
      else:
        raise ValueError(f"Could not read image dimensions from {image_path}")
    
    image_info = {
      "id": self.image_id,
      "file_name": os.path.basename(image_path),
      "width": width,
      "height": height
    }
    
    self.coco_data["images"].append(image_info)
    current_id = self.image_id
    self.image_id += 1
    return current_id
  
  def add_annotation(self, image_id, category_id, bbox, keypoints=None, confidence=None):
    """
    Add annotation to COCO data
    
    Args:
      image_id (int): ID of the image this annotation belongs to
      category_id (int): Category ID
      bbox (list): Bounding box [x, y, width, height]
      keypoints (list): List of keypoint coordinates [x1, y1, v1, x2, y2, v2, ...]
      confidence (float): Detection confidence
    """
    annotation = {
      "id": self.annotation_id,
      "image_id": image_id,
      "category_id": category_id,
      "bbox": bbox,
      "area": bbox[2] * bbox[3],  # width * height
      "iscrowd": 0
    }
    
    if keypoints is not None:
      annotation["keypoints"] = keypoints
      # Count visible keypoints (visibility > 0)
      num_keypoints = sum(1 for i in range(2, len(keypoints), 3) if keypoints[i] > 0)
      annotation["num_keypoints"] = num_keypoints
    
    if confidence is not None:
      annotation["confidence"] = confidence
    
    self.coco_data["annotations"].append(annotation)
    self.annotation_id += 1
  
  def process_yolo_results(self, results, image_id):
    """
    Process YOLO results and add annotations
    
    Args:
      results: YOLO prediction results
      image_id (int): ID of the image
    """
    for result in results:
      if result.boxes is not None:
        boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        
        keypoints_data = None
        if result.keypoints is not None:
          keypoints_data = result.keypoints.data.cpu().numpy()  # [n, num_keypoints, 3]
        
        for i, (box, conf, cls_id) in enumerate(zip(boxes, confidences, class_ids)):
          # Convert box format from x1,y1,x2,y2 to x,y,w,h
          x1, y1, x2, y2 = box
          bbox = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
          
          # Process keypoints if available
          keypoints = None
          if keypoints_data is not None and i < len(keypoints_data):
            kp = keypoints_data[i]  # [num_keypoints, 3]
            keypoints = []
            for j in range(len(kp)):
              x, y, conf_kp = kp[j]
              # COCO format: [x, y, visibility]
              # visibility: 0=not labeled, 1=labeled but not visible, 2=labeled and visible
              visibility = 2 if conf_kp > 0.5 else 1 if conf_kp > 0 else 0
              keypoints.extend([float(x), float(y), visibility])
          
          self.add_annotation(
            image_id=image_id,
            category_id=int(cls_id) + 1,  # Convert YOLO 0-based to COCO 1-based (0->1, 1->2, 2->3)
            bbox=bbox,
            keypoints=keypoints,
            confidence=float(conf)
          )
  
  def save_json(self, output_path):
    """
    Save COCO data to JSON file
    
    Args:
      output_path (str): Path to save the JSON file
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
      json.dump(self.coco_data, f, indent=2, ensure_ascii=False)
    
    print(f"COCO annotations saved to: {output_path}")
    print(f"Total images: {len(self.coco_data['images'])}")
    print(f"Total annotations: {len(self.coco_data['annotations'])}")
  
  def get_statistics(self):
    """Get statistics about the dataset"""
    stats = {
      "total_images": len(self.coco_data["images"]),
      "total_annotations": len(self.coco_data["annotations"]),
      "categories": len(self.coco_data["categories"])
    }
    
    # Count annotations per category
    category_counts = {}
    for ann in self.coco_data["annotations"]:
      cat_id = ann["category_id"]
      category_counts[cat_id] = category_counts.get(cat_id, 0) + 1
    
    stats["annotations_per_category"] = category_counts
    
    return stats 