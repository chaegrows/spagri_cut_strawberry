from src import load_yolo_keypoint_model, predict_keypoints, visualize_keypoints
import argparse
import os

def main(args):
    device = args.device
    if args.device_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_id)
    print(f"Using device: {device}")
    
    model = load_yolo_keypoint_model(args.model_path, device=device)
    results = predict_keypoints(model, args.image_path, device=device, conf_threshold=args.conf_threshold)
    visualize_keypoints(results, args.output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="straw_keypoint.pt")
    parser.add_argument("--image_path", type=str, default="294.png")
    parser.add_argument("--output_path", type=str, default="output.jpg")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--device_id", type=int, default=2)
    parser.add_argument("--conf_threshold", type=float, default=0.01)
    args = parser.parse_args()
    main(args)