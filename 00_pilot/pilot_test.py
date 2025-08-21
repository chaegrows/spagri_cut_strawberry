import os
import cv2
from ultralytics import SAM
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# 모델 로드
model = SAM("mobile_sam.pt")

# 이미지 폴더
img_dir = "spagri_day"

# 클릭 좌표 저장
click_points = []

def mouse_callback(event, x, y, flags, param):
    global click_points
    if event == cv2.EVENT_LBUTTONDOWN:  # 왼쪽 클릭 → positive
        click_points.append(([x, y], 1))
        print(f"Positive point: ({x}, {y})")
    elif event == cv2.EVENT_RBUTTONDOWN:  # 오른쪽 클릭 → negative
        click_points.append(([x, y], 0))
        print(f"Negative point: ({x}, {y})")

def process_image(img_path):
    global click_points
    click_points = []  # 초기화
    img = cv2.imread(img_path)

    # 윈도우와 마우스 이벤트 등록
    cv2.namedWindow("Image")
    cv2.setMouseCallback("Image", mouse_callback)

    print("왼쪽 클릭=positive, 오른쪽 클릭=negative, Enter=실행, ESC=다음 이미지")

    while True:
        # 포인트 표시된 상태로 보여주기
        temp = img.copy()
        for (pt, label) in click_points:
            color = (0, 255, 0) if label == 1 else (0, 0, 255)
            cv2.circle(temp, pt, 5, color, -1)
        cv2.imshow("Image", temp)

        key = cv2.waitKey(1) & 0xFF
        if key == 13:  # Enter
            if click_points:
                points = [pt for pt, _ in click_points]
                labels = [lb for _, lb in click_points]
                results = model.predict(img_path, points=[points], labels=[labels])
                results[0].save(filename=f"./runs/segment/{os.path.basename(img_path)}")
                cv2.imwrite(f"./runs/segment/{os.path.basename(img_path)}", results[0].plot())
                print("Segmentation done!")
            break
        elif key == 27:  # ESC → skip
            break

    cv2.destroyAllWindows()

# 폴더 순회
for filename in os.listdir(img_dir):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        process_image(os.path.join(img_dir, filename))
