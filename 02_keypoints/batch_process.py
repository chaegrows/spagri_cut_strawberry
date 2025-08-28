import os
import argparse
from pathlib import Path
import glob
from tqdm import tqdm
import json

from src import load_yolo_keypoint_model, predict_keypoints
from src.coco_converter import COCOConverter

def get_image_files(folder_path, extensions=None):
  """
  Get all image files from a folder
  
  Args:
    folder_path (str): Path to the folder containing images
    extensions (list): List of image extensions to search for
  
  Returns:
    list: List of image file paths
  """
  if extensions is None:
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
  
  image_files = []
  folder_path = Path(folder_path)
  
  for ext in extensions:
    image_files.extend(glob.glob(str(folder_path / ext), recursive=False))
    image_files.extend(glob.glob(str(folder_path / ext.upper()), recursive=False))
  
  return sorted(image_files)

def setup_output_structure(output_dir, dataset_name="strawberry_keypoints"):
  """
  Setup output directory structure following COCO dataset format
  
  Args:
    output_dir (str): Base output directory
    dataset_name (str): Name of the dataset
  
  Returns:
    dict: Dictionary with paths to different components
  """
  base_path = Path(output_dir) / dataset_name
  
  paths = {
    'base': base_path,
    'images': base_path / 'images',
    'annotations': base_path / 'annotations'
  }
  
  # Create directories
  for path in paths.values():
    path.mkdir(parents=True, exist_ok=True)
  
  return paths

def process_images_batch(model, image_files, output_paths, device='cpu', conf_threshold=0.01, 
                        categories=None, split_name='train'):
  """
  Process a batch of images and save results in COCO format
  
  Args:
    model: Loaded YOLO model
    image_files (list): List of image file paths
    output_paths (dict): Dictionary with output paths
    device (str): Device for inference
    conf_threshold (float): Confidence threshold
    categories (list): Category configuration
    split_name (str): Dataset split name (train/val/test)
  """
  # Initialize COCO converter
  converter = COCOConverter(categories=categories)
  
  print(f"Processing {len(image_files)} images...")
  
  # Process each image
  for image_path in tqdm(image_files, desc="Processing images"):
    try:
      # Add image to COCO data
      image_id = converter.add_image(image_path)
      
      # Run inference
      results = predict_keypoints(model, image_path, conf_threshold=conf_threshold, device=device)
      
      if results:
        # Process YOLO results and add annotations
        converter.process_yolo_results(results, image_id)
      
    except Exception as e:
      print(f"Error processing {image_path}: {e}")
      continue
  
  # Save annotations
  annotation_file = output_paths['annotations'] / f'instances_{split_name}.json'
  converter.save_json(str(annotation_file))
  
  # Print statistics
  stats = converter.get_statistics()
  print(f"\nDataset Statistics:")
  print(f"Split: {split_name}")
  for key, value in stats.items():
    print(f"  {key}: {value}")
  
  return converter, stats

def copy_images_to_output(image_files, output_images_dir, split_name='train'):
  """
  Copy images to output directory structure
  
  Args:
    image_files (list): List of source image paths
    output_images_dir (Path): Output images directory
    split_name (str): Dataset split name
  """
  import shutil
  
  split_dir = output_images_dir / split_name
  split_dir.mkdir(exist_ok=True)
  
  print(f"Copying {len(image_files)} images to {split_dir}...")
  
  for image_path in tqdm(image_files, desc="Copying images"):
    src_path = Path(image_path)
    dst_path = split_dir / src_path.name
    
    try:
      shutil.copy2(src_path, dst_path)
    except Exception as e:
      print(f"Error copying {image_path}: {e}")

def create_dataset_info(output_paths, stats, categories):
  """
  Create dataset information file
  
  Args:
    output_paths (dict): Dictionary with output paths
    stats (dict): Dataset statistics
    categories (list): Category configuration
  """
  info = {
    "dataset_name": "SPAGRI Strawberry Keypoints",
    "description": "Strawberry keypoint detection dataset processed with YOLO",
    "categories": categories,
    "statistics": stats,
    "format": "COCO",
    "created_by": "SPAGRI batch processing script"
  }
  
  info_file = output_paths['base'] / 'dataset_info.json'
  with open(info_file, 'w', encoding='utf-8') as f:
    json.dump(info, f, indent=2, ensure_ascii=False)
  
  print(f"Dataset info saved to: {info_file}")

def main(args):
  """Main processing function"""
  
  # Setup device
  device = args.device
  if args.device_id is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_id)
  print(f"Using device: {device}")
  
  # Load model
  print("Loading YOLO model...")
  model = load_yolo_keypoint_model(args.model_path, device=device)
  if model is None:
    print("Failed to load model. Exiting.")
    return
  
  # Get image files
  print(f"Scanning for images in: {args.input_folder}")
  image_files = get_image_files(args.input_folder)
  
  if not image_files:
    print("No image files found in the specified folder.")
    return
  
  print(f"Found {len(image_files)} image files")
  
  # Setup output structure
  output_paths = setup_output_structure(args.output_dir, args.dataset_name)
  print(f"Output will be saved to: {output_paths['base']}")
  
  # Define categories based on actual model output
  # Model classes: {0: 'partripe', 1: 'ripe', 2: 'unripe'}
  categories = [
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
  
  # Split images if requested
  if args.split_ratio > 0 and args.split_ratio < 1:
    import random
    random.seed(42)  # For reproducible splits
    
    shuffled_files = image_files.copy()
    random.shuffle(shuffled_files)
    
    split_idx = int(len(shuffled_files) * args.split_ratio)
    train_files = shuffled_files[:split_idx]
    val_files = shuffled_files[split_idx:]
    
    print(f"Split dataset: {len(train_files)} train, {len(val_files)} val")
    
    # Process train split
    if train_files:
      print("\nProcessing training set...")
      train_converter, train_stats = process_images_batch(
        model, train_files, output_paths, device, args.conf_threshold, categories, 'train'
      )
      
      if args.copy_images:
        copy_images_to_output(train_files, output_paths['images'], 'train')
    
    # Process validation split
    if val_files:
      print("\nProcessing validation set...")
      val_converter, val_stats = process_images_batch(
        model, val_files, output_paths, device, args.conf_threshold, categories, 'val'
      )
      
      if args.copy_images:
        copy_images_to_output(val_files, output_paths['images'], 'val')
      
      # Combine statistics
      combined_stats = {
        'train': train_stats,
        'val': val_stats,
        'total_images': train_stats['total_images'] + val_stats['total_images'],
        'total_annotations': train_stats['total_annotations'] + val_stats['total_annotations']
      }
  
  else:
    # Process all images as single split
    print(f"\nProcessing all images as '{args.split_name}' split...")
    converter, stats = process_images_batch(
      model, image_files, output_paths, device, args.conf_threshold, categories, args.split_name
    )
    
    if args.copy_images:
      copy_images_to_output(image_files, output_paths['images'], args.split_name)
    
    combined_stats = {args.split_name: stats}
  
  # Create dataset info
  create_dataset_info(output_paths, combined_stats, categories)
  
  print(f"\nProcessing completed! Results saved to: {output_paths['base']}")

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Batch process images with YOLO keypoint detection and save as COCO format")
  
  # Input/Output arguments
  parser.add_argument("--input_folder", type=str, required=True,
                     help="Path to folder containing input images")
  parser.add_argument("--output_dir", type=str, default="./dataset_output",
                     help="Base output directory for processed dataset")
  parser.add_argument("--dataset_name", type=str, default="strawberry_keypoints",
                     help="Name of the output dataset")
  
  # Model arguments
  parser.add_argument("--model_path", type=str, default="straw_keypoint.pt",
                     help="Path to YOLO model weights")
  parser.add_argument("--device", type=str, default="cuda",
                     help="Device for inference (cuda/cpu)")
  parser.add_argument("--device_id", type=int, default=2,
                     help="CUDA device ID")
  parser.add_argument("--conf_threshold", type=float, default=0.01,
                     help="Confidence threshold for detections")
  
  # Dataset split arguments
  parser.add_argument("--split_ratio", type=float, default=0.0,
                     help="Ratio for train/val split (0.0 = no split, 0.8 = 80% train, 20% val)")
  parser.add_argument("--split_name", type=str, default="train",
                     help="Split name when not using train/val split")
  
  # Processing options
  parser.add_argument("--copy_images", action="store_true",
                     help="Copy images to output directory structure")
  
  args = parser.parse_args()
  main(args) 