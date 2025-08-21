import torch
from ultralytics import YOLO
import cv2
import numpy as np

def load_yolo_keypoint_model(weight_path, device='cpu'):
  """
  Load YOLOv11 keypoint detection model from weight file
  
  Args:
    weight_path (str): Path to the weight file (.pt)
    device (str): Device to load model on ('cpu' or 'cuda')
  
  Returns:
    YOLO: Loaded YOLO model
  """
  try:
    # Load the model
    model = YOLO(weight_path)
    
    # Move model to specified device
    model.to(device)
    print(f"Model loaded successfully from: {weight_path}")
    print(f"Device: {device}")
    
    # Print model information
    print(f"Model type: {model.task}")
    print(f"Model names: {model.names}")
    
    return model
  
  except Exception as e:
    print(f"Error loading model: {e}")
    return None

def predict_keypoints(model, image_path, conf_threshold=0.5, device='cpu'):
  """
  Predict keypoints on an image
  
  Args:
    model (YOLO): Loaded YOLO model
    image_path (str): Path to input image
    conf_threshold (float): Confidence threshold
    device (str): Device for inference ('cpu' or 'cuda')
  
  Returns:
    Results: YOLO prediction results
  """
  try:
    # Run inference with device specification
    results = model(image_path, conf=conf_threshold, device=device)
    
    # Print detection info
    for result in results:
      if result.keypoints is not None:
        print(f"Detected {len(result.keypoints.data)} objects with keypoints")
        print(f"Keypoint shape: {result.keypoints.data.shape}")
      else:
        print("No keypoints detected")
    
    return results
  
  except Exception as e:
    print(f"Error during prediction: {e}")
    return None

def visualize_keypoints(results, save_path=None):
  """
  Visualize keypoint detection results
  
  Args:
    results: YOLO prediction results
    save_path (str): Path to save the visualization
  """
  try:
    for result in results:
      # Plot the results
      annotated_frame = result.plot()
      
      if save_path:
        cv2.imwrite(save_path, annotated_frame)
        print(f"Visualization saved to: {save_path}")
      else:
        # Display the image
        cv2.imshow('YOLOv11 Keypoint Detection', annotated_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
  
  except Exception as e:
    print(f"Error during visualization: {e}")

# Main execution example
if __name__ == "__main__":
  # Check if CUDA is available and compatible
  device = 'cpu'  # Default to CPU to avoid CUDA compatibility issues
  
  try:
    import torch
    if torch.cuda.is_available():
      print("CUDA is available but using CPU for compatibility")
    else:
      print("CUDA not available, using CPU")
  except:
    print("Using CPU device")
  
  # Load the model
  weight_path = "weight.pt"  # Path to your weight file
  model = load_yolo_keypoint_model(weight_path, device=device)
  
  if model is not None:
    # Example usage with an image
    # image_path = "path/to/your/image.jpg"
    # results = predict_keypoints(model, image_path, device=device)
    # 
    # if results:
    #   visualize_keypoints(results, "output.jpg")
    
    print("Model is ready for inference!")
    print("Uncomment the example code above to test with an image.")
    print(f"Model will run on: {device}")
  else:
    print("Failed to load the model.")

