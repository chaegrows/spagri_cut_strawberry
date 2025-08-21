import argparse
import glob
import os
from pathlib import Path

import cv2
import numpy as np
from ultralytics import SAM
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# -------- Args --------
parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, required=True,
                    help="이미지 파일 또는 이미지 폴더 경로")
parser.add_argument("--weights", type=str, default="mobile_sam.pt",
                    help="Mobile-SAM 가중치 파일 경로")
parser.add_argument("--out", type=str, default="sam_out",
                    help="결과 저장 폴더")
parser.add_argument("--device", type=str, default="", help="'cuda:0' 또는 'cpu' 지정 가능")
parser.add_argument("--alpha", type=float, default=0.5, help="마스크 오버레이 투명도(0~1)")
args = parser.parse_args()
import math

def mask_to_aabb(mask: np.ndarray):
    """타이트한 axis-aligned bbox 반환: (x1,y1,x2,y2)"""
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    return [x1, y1, x2, y2]

def mask_to_rbox(mask: np.ndarray):
    """
    회전 사각형: (box_points(4,2) int), rect=(cx,cy),(w,h),angle(deg)
    angle: OpenCV 규약(대개 -90~0) 주의
    """
    m8 = (mask.astype(np.uint8) * 255)
    cnts, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, None
    # 가장 큰 컨투어 사용
    c = max(cnts, key=cv2.contourArea)
    rect = cv2.minAreaRect(c)  # ((cx,cy),(w,h),angle)
    box = cv2.boxPoints(rect)  # 4x2 float
    box = np.int32(np.round(box))
    return box, rect

def mask_principal_axis(mask: np.ndarray):
    """
    주성분축: 중심(cx,cy), 주축 단위벡터 v(2,), 고유값(λ1, λ2)
    """
    ys, xs = np.where(mask)
    if len(xs) < 5:
        return None
    pts = np.vstack([xs, ys]).T.astype(np.float32)  # (N,2)

    # 중심화
    mean = pts.mean(axis=0)
    C = np.cov((pts - mean).T)
    evals, evecs = np.linalg.eig(C)  # evecs[:,i]가 i번째 고유벡터
    i_max = np.argmax(evals)
    v = evecs[:, i_max]  # (2,)
    v = v / (np.linalg.norm(v) + 1e-8)

    return (float(mean[0]), float(mean[1])), (float(v[0]), float(v[1])), (float(evals[i_max]), float(evals[1 - i_max]))

def draw_principal_axis(img: np.ndarray, center, v, scale=100.0, color=(255, 0, 0), thickness=2):
    """
    중심에서 양방향으로 선 그리기. scale은 길이(픽셀).
    """
    cx, cy = center
    vx, vy = v
    p1 = (int(round(cx - vx * scale)), int(round(cy - vy * scale)))
    p2 = (int(round(cx + vx * scale)), int(round(cy + vy * scale)))
    cv2.line(img, p1, p2, color, thickness, cv2.LINE_AA)
    # 중심점
    cv2.circle(img, (int(round(cx)), int(round(cy))), 3, color, -1, cv2.LINE_AA)

def overlay_rbox(img: np.ndarray, box_pts: np.ndarray, color=(0, 0, 255), thickness=2):
    cv2.polylines(img, [box_pts], isClosed=True, color=color, thickness=thickness, lineType=cv2.LINE_AA)

# -------- 이미지 목록 수집 --------
in_path = Path(args.input)
if in_path.is_dir():
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")
    img_files = []
    for e in exts:
        img_files.extend(sorted(glob.glob(str(in_path / e))))
else:
    img_files = [str(in_path)]

assert len(img_files) > 0, f"No images found under {in_path}"

# -------- 모델 로드 --------
model = SAM(args.weights)
# 참고: SAM은 predict에서 bboxes/points/labels 프롬프트를 받습니다. (xyxy)  # docs ref

# -------- 유틸 --------
def clamp(v, lo, hi):
    return max(lo, min(int(v), hi))

def overlay_mask(image_bgr, mask_bool, alpha=0.5):
    """mask_bool: HxW(bool)"""
    color = np.array([0, 255, 0], dtype=np.uint8)  # 오버레이 색 (초록)
    overlay = image_bgr.copy()
    overlay[mask_bool] = (overlay[mask_bool] * (1 - alpha) + color * alpha).astype(np.uint8)
    return overlay

# -------- 마우스 콜백 상태 --------
class BoxDrawer:
    def __init__(self, img_shape):
        self.h, self.w = img_shape[:2]
        self.reset()

    def reset(self):
        self.drawing = False
        self.x1 = self.y1 = self.x2 = self.y2 = -1
        self.has_box = False

    def on_mouse(self, event, x, y, flags, param):
        x = clamp(x, 0, self.w - 1)
        y = clamp(y, 0, self.h - 1)
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.x1, self.y1 = x, y
            self.x2, self.y2 = x, y
            self.has_box = False
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.x2, self.y2 = x, y
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.x2, self.y2 = x, y
            if abs(self.x2 - self.x1) > 2 and abs(self.y2 - self.y1) > 2:
                self.has_box = True
            else:
                self.reset()

    def get_xyxy(self):
        x1 = min(self.x1, self.x2); y1 = min(self.y1, self.y2)
        x2 = max(self.x1, self.x2); y2 = max(self.y1, self.y2)
        return [x1, y1, x2, y2]

# -------- 메인 루프 --------
os.makedirs(args.out, exist_ok=True)
print("[키 가이드] 드래그: 박스 프롬프트  |  'r': 초기화  |  's': 결과 저장  |  'n': 다음  |  'q': 종료")

for idx, fpath in enumerate(img_files, 1):
    img = cv2.imread(fpath)
    if img is None:
        print(f"[WARN] Fail to read: {fpath}")
        continue

    win = "Mobile-SAM (bbox prompt)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    drawer = BoxDrawer(img.shape)
    cv2.setMouseCallback(win, drawer.on_mouse)

    base = img.copy()
    vis = base.copy()
    last_mask = None
    last_box = None

    while True:
        # 박스 그리기 미리보기
        disp = vis.copy()
        if drawer.drawing or drawer.has_box:
            x1, y1, x2, y2 = drawer.get_xyxy()
            cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 255), 2)

        cv2.imshow(win, disp)
        key = cv2.waitKey(16) & 0xFF

        if key == ord('q'):
            cv2.destroyAllWindows()
            raise SystemExit

        if key == ord('r'):
            drawer.reset()
            vis = base.copy()
            last_mask = None
            last_box = None

        # 마우스에서 박스가 확정되면 예측 수행
        if drawer.has_box:
            xyxy = drawer.get_xyxy()
            last_box = xyxy

            # SAM 예측: 박스 프롬프트 (xyxy)
            results = model.predict(
                source=base,               # ndarray 직접 전달 가능
                bboxes=[xyxy],             # ★ 핵심: 박스 프롬프트
                device=args.device,
                verbose=False
            )
            # 보통 한 결과(r) 내에 1개 마스크가 생성됨
            r = results[0]
            print(r)
            if r.masks is not None and r.masks.data is not None:
                # r.masks.data: (N, H, W) torch.bool/float
                mask = r.masks.data[0].cpu().numpy().astype(bool)
                last_mask = mask
                
                vis = overlay_mask(base, mask, alpha=args.alpha)
                # # --- 타이트 AABB ---
                # aabb = mask_to_aabb(mask)
                # if aabb is not None:
                #     x1, y1, x2, y2 = aabb
                #     cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)  # 노랑 AABB
                #     # 원하면 저장용으로 last_box도 이것으로 교체:
                #     last_box = [x1, y1, x2, y2]

                # # --- 회전 사각형(더 fit하게) ---
                # rbox, rect = mask_to_rbox(mask)
                # if rbox is not None:
                #     overlay_rbox(vis, rbox, color=(0, 0, 255), thickness=2)  # 빨강 RBox
                #     # 각도/중심/가로세로 표시(옵션)
                #     (cx, cy), (w, h), angle = rect
                #     txt = f"RBox angle={angle:.1f}°, w={w:.1f}, h={h:.1f}"
                #     cv2.putText(vis, txt, (int(cx), int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)

                # --- 주성분축(방향) ---
                pax = mask_principal_axis(mask)
                if pax is not None:
                    center, v, evals = pax
                    # 스케일은 마스크 크기에 비례하도록 설정(대략 대각선 길이의 0.5배)
                    H, W = mask.shape
                    scale = 0.5 * math.hypot(W, H)
                    draw_principal_axis(vis, center, v, scale=scale, color=(255, 0, 0), thickness=2)
            else:
                print("[INFO] No mask returned for the box. Try adjusting the box.")
            drawer.has_box = False  # 한 번 처리 후 off

        if key == ord('s'):
            # 저장
            out_name = Path(fpath).stem
            save_dir = Path(args.out)
            save_dir.mkdir(parents=True, exist_ok=True)
            
            cv2.rectangle(vis, (last_box[0], last_box[1]), (last_box[2], last_box[3]), (0, 0, 255), 2)
            cv2.imwrite(str(save_dir / f"{out_name}_overlay.png"), vis)
            if last_mask is not None:
                # 마스크를 0/255 PNG로 저장
                mask_u8 = (last_mask.astype(np.uint8) * 255)
                cv2.imwrite(str(save_dir / f"{out_name}_mask.png"), mask_u8)
            if last_box is not None:
                with open(save_dir / f"{out_name}_box.txt", "w") as fw:
                    fw.write(",".join(map(str, last_box)))
            print(f"[SAVE] {save_dir}")

        if key == ord('n'):
            break  # 다음 이미지로

    cv2.destroyAllWindows()
# # 폴더 전체
# python sam_box_click_segment.py --input /path/to/images --out sam_out
# python method3.py --input /media/metafarmers/bag1/SPAGRI/pilot/spagri_day --out sam_out

# # 단일 이미지
# python method3.py --input /media/metafarmers/bag1/SPAGRI/pilot/spagri_day/straw_seg_000008.jpg --out sam_out

# # GPU 지정 (가능하면)
# python sam_box_click_segment.py --input ./imgs --device cuda:0