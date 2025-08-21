from model_load import load_yolo_keypoint_model, predict_keypoints, visualize_keypoints

# 모델 로드
model = load_yolo_keypoint_model("weight.pt")

# 이미지에서 keypoint 예측
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
results = predict_keypoints(model, "294.png")

# 결과 시각화
visualize_keypoints(results, "output.jpg")