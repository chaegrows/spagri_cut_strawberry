#!/usr/bin/env python3
"""
Example usage of batch processing script for YOLO keypoint detection
Updated with correct COCO category format for strawberry ripeness detection
"""

import subprocess
import os
from pathlib import Path

def run_batch_processing():
  """Example of running batch processing with different configurations"""
  
  # Base paths - adjust these according to your setup
  input_folder = "/media/metafarmers/bag1/SPAGRI/02_keypoints/evaluation_results"  # Folder with images
  output_dir = "/media/metafarmers/bag1/SPAGRI/02_keypoints/dataset_output"
  model_path = "/media/metafarmers/bag1/SPAGRI/02_keypoints/straw_keypoint.pt"
  
  # Example 1: Process all images as single dataset
  print("Example 1: Processing all images as single dataset")
  cmd1 = [
    "python", "batch_process.py",
    "--input_folder", input_folder,
    "--output_dir", output_dir,
    "--dataset_name", "strawberry_single",
    "--model_path", model_path,
    "--device", "cuda",
    "--device_id", "2",
    "--conf_threshold", "0.01",
    "--split_name", "train",
    "--copy_images"
  ]
  
  # Example 2: Process with train/val split
  print("\nExample 2: Processing with 80/20 train/val split")
  cmd2 = [
    "python", "batch_process.py",
    "--input_folder", input_folder,
    "--output_dir", output_dir,
    "--dataset_name", "strawberry_split",
    "--model_path", model_path,
    "--device", "cuda",
    "--device_id", "2",
    "--conf_threshold", "0.01",
    "--split_ratio", "0.8",
    "--copy_images"
  ]
  
  # Example 3: Process without copying images (annotations only)
  print("\nExample 3: Processing annotations only (no image copying)")
  cmd3 = [
    "python", "batch_process.py",
    "--input_folder", input_folder,
    "--output_dir", output_dir,
    "--dataset_name", "strawberry_annotations_only",
    "--model_path", model_path,
    "--device", "cpu",  # Use CPU if GPU is not available
    "--conf_threshold", "0.05",
    "--split_name", "test"
  ]
  
  # Choose which example to run
  examples = {
    "1": cmd1,
    "2": cmd2,
    "3": cmd3
  }
  
  print("\nAvailable examples:")
  print("1. Single dataset (all images as train)")
  print("2. Train/Val split (80/20)")
  print("3. Annotations only (no image copying)")
  print("\nCategory Information:")
  print("Model classes: {0: 'partripe', 1: 'ripe', 2: 'unripe'}")
  print("COCO format:")
  print("- partripe (id: 1): Partially ripe strawberry")
  print("- ripe (id: 2): Fully ripe strawberry") 
  print("- unripe (id: 3): Unripe strawberry")
  print("Keypoints: fruit, calyx, stem")
  
  choice = input("\nSelect example to run (1-3) or 'q' to quit: ").strip()
  
  if choice in examples:
    print(f"\nRunning example {choice}...")
    print(f"Command: {' '.join(examples[choice])}")
    
    try:
      result = subprocess.run(examples[choice], check=True, capture_output=True, text=True)
      print("Success!")
      print(result.stdout)
    except subprocess.CalledProcessError as e:
      print(f"Error running command: {e}")
      print(f"Stderr: {e.stderr}")
      print(f"Stdout: {e.stdout}")
  
  elif choice.lower() == 'q':
    print("Exiting...")
  else:
    print("Invalid choice. Please select 1-3 or 'q'.")

if __name__ == "__main__":
  # Change to script directory
  script_dir = Path(__file__).parent
  os.chdir(script_dir)
  
  print("SPAGRI Strawberry Keypoint Batch Processing")
  print("Strawberry Ripeness Detection with Keypoints")
  print("=" * 50)
  
  # Check if required files exist
  required_files = [
    "batch_process.py",
    "straw_keypoint.pt",
    "src/model_load.py",
    "src/coco_converter.py"
  ]
  
  missing_files = []
  for file in required_files:
    if not Path(file).exists():
      missing_files.append(file)
  
  if missing_files:
    print("Missing required files:")
    for file in missing_files:
      print(f"  - {file}")
    print("\nPlease ensure all required files are present before running.")
  else:
    run_batch_processing() 