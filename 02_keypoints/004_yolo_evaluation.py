#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import csv
from datetime import datetime
from pathlib import Path
import glob

import cv2
import numpy as np
from src.model_load import load_yolo_keypoint_model, predict_keypoints

"""
[YOLO Keypoint Evaluation Tool]
- YOLO 키포인트 결과와 수동 측정 각도를 비교 평가하는 도구
- 한 이미지의 여러 딸기를 순회하며 각 딸기의 kp0-kp1, kp1-kp2 쌍을 평가
- 각 딸기마다 3개 키포인트 중 2개 쌍만 평가

조작:
  - 좌클릭: 수동 측정점 찍기(P1->P2). 두 점이 모이면 각도 계산 & 미리보기
  - c: 현재 측정값 확정 & CSV 저장
  - r: 현재 측정 취소 & 다시 시작
  - t: YOLO 키포인트 표시 토글
  - a/d: 이전/다음 딸기 선택 (같은 이미지 내)
  - 1,2: 키포인트 쌍 선택 (1: kp0-kp1, 2: kp1-kp2)
  - u: 마지막 점 되돌리기
  - s: 현재 화면 저장
  - n/Enter: 다음 이미지
  - p: 이전 이미지
  - k: 건너뛰기
  - q/ESC: 종료
"""

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
CSV_HEADER = [
    "timestamp",
    "image",
    "strawberry_idx",
    "manual_angle_deg",
    "manual_p1_x", "manual_p1_y",
    "manual_p2_x", "manual_p2_y",
    # kp0-kp1 pair data
    "yolo_kp0_kp1_angle_deg",
    "yolo_kp0_kp1_angle_diff_deg",
    "yolo_kp0_p1_x", "yolo_kp0_p1_y",
    "yolo_kp0_p2_x", "yolo_kp0_p2_y",
    "yolo_kp0_kp1_confidence",
    # kp1-kp2 pair data
    "yolo_kp1_kp2_angle_deg",
    "yolo_kp1_kp2_angle_diff_deg",
    "yolo_kp1_p1_x", "yolo_kp1_p1_y",
    "yolo_kp1_p2_x", "yolo_kp1_p2_y",
    "yolo_kp1_kp2_confidence"
]

def draw_hud(img, msg_lines, org=(10, 26), scale=0.6, color=(255, 255, 255)):
    y = org[1]
    for line in msg_lines:
        cv2.putText(img, line, (org[0], y),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, line, (org[0], y),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
        y += int(24 * scale + 8)

def draw_reference_arrow(canvas, p, length=60, color=(0, 200, 255)):
    p = tuple(map(int, p))
    q = (int(p[0]), int(p[1] - length))
    cv2.arrowedLine(canvas, p, q, color, 2, tipLength=0.25)

def compute_angles(p1, p2):
    v = np.array([p2[0] - p1[0], p2[1] - p1[1]], dtype=float)
    if np.allclose(v, 0):
        return None, None
    angle_cw = (np.degrees(np.arctan2(v[0], -v[1])) % 360.0)
    angle_unsigned = min(angle_cw, 360.0 - angle_cw)
    return angle_cw, angle_unsigned

def draw_yolo_keypoints(img, keypoints_data, selected_strawberry=0, selected_pair=0):
    """Draw YOLO keypoints and highlight selected strawberry and pair"""
    overlay = img.copy()

    if keypoints_data is None or len(keypoints_data) == 0:
        return overlay

    # Define keypoint pairs (only kp0-kp1 and kp1-kp2)
    pairs = [(0, 1), (1, 2)]  # kp0-kp1, kp1-kp2
    colors = [(255, 0, 255), (0, 255, 255)]  # Magenta, Cyan

    # Draw all strawberries
    for strawberry_idx, kpts in enumerate(keypoints_data):
        if kpts.shape[0] < 3:
            continue

        # Determine if this strawberry is selected
        is_selected = (strawberry_idx == selected_strawberry)
        alpha = 1.0 if is_selected else 0.4

        # Draw strawberry bounding box or indicator
        if is_selected:
            # Draw a bounding box around selected strawberry keypoints
            valid_kpts = kpts[kpts[:, 2] > 0.5]
            if len(valid_kpts) > 0:
                min_x, min_y = valid_kpts[:, :2].min(axis=0)
                max_x, max_y = valid_kpts[:, :2].max(axis=0)
                cv2.rectangle(overlay, (int(min_x)-20, int(min_y)-20),
                             (int(max_x)+20, int(max_y)+20), (0, 255, 0), 2)
                cv2.putText(overlay, f"Strawberry {strawberry_idx}",
                           (int(min_x)-20, int(min_y)-30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Draw keypoints
        for i, (x, y, conf) in enumerate(kpts):
            if conf > 0.5:  # Only draw confident keypoints
                point_color = (0, 255, 0) if is_selected else (100, 200, 100)
                cv2.circle(overlay, (int(x), int(y)), 8 if is_selected else 5, point_color, -1)
                cv2.putText(overlay, f"kp{i}", (int(x)+10, int(y)-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Draw keypoint pairs
        for i, (p1_idx, p2_idx) in enumerate(pairs):
            if (kpts[p1_idx][2] > 0.5 and kpts[p2_idx][2] > 0.5):  # Both points confident
                p1 = (int(kpts[p1_idx][0]), int(kpts[p1_idx][1]))
                p2 = (int(kpts[p2_idx][0]), int(kpts[p2_idx][1]))

                # Highlight selected pair of selected strawberry
                if is_selected and i == selected_pair:
                    thickness = 4
                    color = colors[i]
                elif is_selected:
                    thickness = 2
                    color = tuple(int(c * 0.7) for c in colors[i])
                else:
                    thickness = 1
                    color = tuple(int(c * 0.3) for c in colors[i])

                cv2.line(overlay, p1, p2, color, thickness)

                # Draw angle for selected pair of selected strawberry
                if is_selected and i == selected_pair:
                    angle_cw, _ = compute_angles(p1, p2)
                    if angle_cw is not None:
                        mid_x = (p1[0] + p2[0]) // 2
                        mid_y = (p1[1] + p2[1]) // 2
                        cv2.putText(overlay, f"YOLO: {angle_cw:.1f}deg",
                                   (mid_x, mid_y-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return overlay

def get_yolo_keypoint_pair(keypoints_data, strawberry_idx, pair_idx):
    """Get specific keypoint pair from YOLO results for a specific strawberry"""
    if keypoints_data is None or len(keypoints_data) == 0:
        return None, None, None

    if strawberry_idx >= len(keypoints_data):
        return None, None, None

    kpts = keypoints_data[strawberry_idx]
    pairs = [(0, 1), (1, 2)]  # Only kp0-kp1 and kp1-kp2

    if pair_idx >= len(pairs) or kpts.shape[0] < 3:
        return None, None, None

    p1_idx, p2_idx = pairs[pair_idx]

    if kpts[p1_idx][2] > 0.5 and kpts[p2_idx][2] > 0.5:
        p1 = (int(kpts[p1_idx][0]), int(kpts[p1_idx][1]))
        p2 = (int(kpts[p2_idx][0]), int(kpts[p2_idx][1]))
        conf = (kpts[p1_idx][2] + kpts[p2_idx][2]) / 2
        return p1, p2, conf

    return None, None, None

def ensure_csv_with_header(csv_path: Path):
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADER)

def append_csv_row(csv_path: Path, row: list):
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)

def list_images(folder: Path):
    files = []
    for ext in IMG_EXTS:
        files.extend(sorted(folder.glob(f"*{ext}")))
    return files

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True, help="Image folder path")
    ap.add_argument("--model", required=True, help="YOLO model path")
    ap.add_argument("--out", default="yolo_evaluation.csv", help="Output CSV path")
    ap.add_argument("--device", default="cpu", help="Device for YOLO inference")
    ap.add_argument("--conf", type=float, default=0.1, help="YOLO confidence threshold")
    args = ap.parse_args()

    folder = Path(args.folder)
    if not folder.exists() or not folder.is_dir():
        raise NotADirectoryError(f"Folder does not exist: {folder}")

    images = list_images(folder)
    if not images:
        raise FileNotFoundError(f"Image not found: {folder} ({IMG_EXTS})")

    # Load YOLO model
    print("Loading YOLO model...")
    model = load_yolo_keypoint_model(args.model, device=args.device)
    if model is None:
        raise RuntimeError("Failed to load YOLO model")

    csv_path = Path(args.out).resolve()
    ensure_csv_with_header(csv_path)

    window = "YOLO Keypoint Evaluation"
    cv2.namedWindow(window, cv2.WINDOW_FULLSCREEN)

    idx = 0
    points = []
    selected_strawberry = 0  # Selected strawberry index in current image
    selected_pair = 0  # 0: kp0-kp1, 1: kp1-kp2
    show_yolo = True
    pending_measurement = None
    yolo_results = {}  # Cache YOLO results

    def load_image_and_predict(i):
        img = cv2.imread(str(images[i]))
        if img is None:
            raise FileNotFoundError(f"Cannot open image: {images[i]}")

        # Get YOLO predictions
        if i not in yolo_results:
            print(f"Running YOLO inference on {images[i].name}...")
            results = predict_keypoints(model, str(images[i]),
                                      conf_threshold=args.conf, device=args.device)
            if results and results[0].keypoints is not None:
                yolo_results[i] = results[0].keypoints.data.cpu().numpy()
            else:
                yolo_results[i] = None

        return img, yolo_results[i]

    img, keypoints_data = load_image_and_predict(idx)
    display = img.copy()

    def update_display():
        nonlocal display
        display = img.copy()

        if show_yolo and keypoints_data is not None:
            display = draw_yolo_keypoints(display, keypoints_data, selected_strawberry, selected_pair)

        pair_names = ["kp0-kp1", "kp1-kp2"]
        yolo_info = "No YOLO detection"
        strawberry_info = "No strawberries detected"

        if keypoints_data is not None:
            num_strawberries = len(keypoints_data)
            strawberry_info = f"Strawberry {selected_strawberry+1}/{num_strawberries}"

            if selected_strawberry < num_strawberries:
                yolo_p1, yolo_p2, yolo_conf = get_yolo_keypoint_pair(keypoints_data, selected_strawberry, selected_pair)
                if yolo_p1 is not None:
                    yolo_angle, _ = compute_angles(yolo_p1, yolo_p2)
                    yolo_info = f"YOLO {pair_names[selected_pair]}: {yolo_angle:.1f}deg (conf: {yolo_conf:.2f})"
                else:
                    yolo_info = f"YOLO {pair_names[selected_pair]}: Low confidence"

        hud = [
            f"[{idx+1}/{len(images)}] {images[idx].name}",
            strawberry_info,
            f"Selected pair: {pair_names[selected_pair]} (Press 1,2 to change)",
            yolo_info,
            "Left click: manual measurement; c: confirm & save; r: reset",
            "a/d: prev/next strawberry; t: toggle YOLO display; u: undo; s: save screen",
            "n/Enter: next, p: prev, k: skip, q/ESC: quit"
        ]

        if pending_measurement:
            hud.append(f"Manual: {pending_measurement['angle_cw']:.1f}deg (Press 'c' to save)")

        draw_hud(display, hud)
        cv2.imshow(window, display)

    update_display()
    cv2.waitKey(150)

    def on_mouse(event, x, y, flags, param):
        nonlocal points, pending_measurement, display

        if event == cv2.EVENT_LBUTTONDOWN:
            if len(points) < 2:
                points.append((x, y))

            if len(points) == 2:
                p1, p2 = points
                overlay = display.copy()

                # Draw manual measurement
                cv2.circle(overlay, p1, 5, (0, 255, 0), -1, cv2.LINE_AA)
                cv2.circle(overlay, p2, 5, (0, 0, 255), -1, cv2.LINE_AA)
                cv2.line(overlay, p1, p2, (255, 0, 0), 2, cv2.LINE_AA)

                # Reference arrow
                draw_reference_arrow(overlay, p1, length=60)

                # Calculate angle
                angle_cw, angle_unsigned = compute_angles(p1, p2)

                if angle_cw is None:
                    return

                # Store pending measurement
                ts = datetime.now().isoformat(timespec="seconds")
                pending_measurement = {
                    'ts': ts,
                    'p1': p1,
                    'p2': p2,
                    'angle_cw': angle_cw,
                    'angle_unsigned': angle_unsigned,
                    'overlay': overlay
                }

                # Draw manual angle text
                txt = f"Manual: {angle_cw:.1f}deg"
                txt_org = (p1[0] + 10, p1[1] - 10)
                cv2.putText(overlay, txt, txt_org,
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(overlay, txt, txt_org,
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

                display = overlay
                points.clear()
                update_display()

    cv2.setMouseCallback(window, on_mouse)

    def save_measurement():
        nonlocal pending_measurement

        if pending_measurement is None:
            return

        # Get both YOLO keypoint pairs
        yolo_kp0_p1, yolo_kp0_p2, yolo_kp0_conf = get_yolo_keypoint_pair(keypoints_data, selected_strawberry, 0)  # kp0-kp1
        yolo_kp1_p1, yolo_kp1_p2, yolo_kp1_conf = get_yolo_keypoint_pair(keypoints_data, selected_strawberry, 1)  # kp1-kp2

        manual_angle = pending_measurement['angle_cw']

        # Calculate angles and differences for both pairs
        yolo_kp0_angle = None
        yolo_kp0_diff = None
        if yolo_kp0_p1 is not None:
            yolo_kp0_angle, _ = compute_angles(yolo_kp0_p1, yolo_kp0_p2)
            yolo_kp0_diff = abs(yolo_kp0_angle - manual_angle)
            if yolo_kp0_diff > 180:
                yolo_kp0_diff = 360 - yolo_kp0_diff
        else:
            yolo_kp0_conf = 0
            yolo_kp0_p1 = (0, 0)
            yolo_kp0_p2 = (0, 0)

        yolo_kp1_angle = None
        yolo_kp1_diff = None
        if yolo_kp1_p1 is not None:
            yolo_kp1_angle, _ = compute_angles(yolo_kp1_p1, yolo_kp1_p2)
            yolo_kp1_diff = abs(yolo_kp1_angle - manual_angle)
            if yolo_kp1_diff > 180:
                yolo_kp1_diff = 360 - yolo_kp1_diff
        else:
            yolo_kp1_conf = 0
            yolo_kp1_p1 = (0, 0)
            yolo_kp1_p2 = (0, 0)

        # Save to CSV with both pairs
        row = [
            pending_measurement['ts'],
            images[idx].name,
            selected_strawberry,
            f"{manual_angle:.6f}",
            pending_measurement['p1'][0], pending_measurement['p1'][1],
            pending_measurement['p2'][0], pending_measurement['p2'][1],
            # kp0-kp1 data
            f"{yolo_kp0_angle:.6f}" if yolo_kp0_angle is not None else "N/A",
            f"{yolo_kp0_diff:.6f}" if yolo_kp0_diff is not None else "N/A",
            yolo_kp0_p1[0], yolo_kp0_p1[1],
            yolo_kp0_p2[0], yolo_kp0_p2[1],
            f"{yolo_kp0_conf:.3f}",
            # kp1-kp2 data
            f"{yolo_kp1_angle:.6f}" if yolo_kp1_angle is not None else "N/A",
            f"{yolo_kp1_diff:.6f}" if yolo_kp1_diff is not None else "N/A",
            yolo_kp1_p1[0], yolo_kp1_p1[1],
            yolo_kp1_p2[0], yolo_kp1_p2[1],
            f"{yolo_kp1_conf:.3f}"
        ]
        append_csv_row(csv_path, row)

        # Save annotated image
        out_dir = Path("evaluation_results") / images[idx].stem
        out_dir.mkdir(parents=True, exist_ok=True)
        pair_names = ["kp0-kp1", "kp1-kp2"]
        out_path = out_dir / f"s{selected_strawberry}_{pair_names[selected_pair]}_{datetime.now().strftime('%H%M%S')}.png"
        cv2.imwrite(str(out_path), pending_measurement['overlay'])

        # Print comparison for both pairs
        kp0_info = f"kp0-kp1: {yolo_kp0_angle:.1f}° (diff: {yolo_kp0_diff:.1f}°)" if yolo_kp0_angle is not None else "kp0-kp1: N/A"
        kp1_info = f"kp1-kp2: {yolo_kp1_angle:.1f}° (diff: {yolo_kp1_diff:.1f}°)" if yolo_kp1_angle is not None else "kp1-kp2: N/A"
        print(f"Saved: Strawberry {selected_strawberry} Manual: {manual_angle:.1f}° vs YOLO - {kp0_info}, {kp1_info}")

        pending_measurement = None
        update_display()

    def reset_measurement():
        nonlocal pending_measurement, points
        pending_measurement = None
        points.clear()
        update_display()

    def reset_strawberry_selection():
        """Reset strawberry selection when changing images"""
        nonlocal selected_strawberry, selected_pair
        selected_strawberry = 0
        selected_pair = 0
        reset_measurement()

    while True:
        key = cv2.waitKey(20) & 0xFF

        if key in (27, ord('q')):  # ESC / q
            break

        elif key == ord('c'):  # confirm and save
            save_measurement()

        elif key == ord('r'):  # reset current measurement
            reset_measurement()

        elif key == ord('t'):  # toggle YOLO display
            show_yolo = not show_yolo
            update_display()

        elif key == ord('a'):  # Previous strawberry
            if keypoints_data is not None and selected_strawberry > 0:
                selected_strawberry -= 1
                reset_measurement()

        elif key == ord('d'):  # Next strawberry
            if keypoints_data is not None and selected_strawberry < len(keypoints_data) - 1:
                selected_strawberry += 1
                reset_measurement()

        elif key == ord('1'):  # Select kp0-kp1 pair
            selected_pair = 0
            reset_measurement()

        elif key == ord('2'):  # Select kp1-kp2 pair
            selected_pair = 1
            reset_measurement()

        elif key in (ord('\r'), ord('\n'), ord('n')):  # Enter / n
            if idx < len(images) - 1:
                idx += 1
                img, keypoints_data = load_image_and_predict(idx)
                reset_strawberry_selection()

        elif key == ord('p'):
            if idx > 0:
                idx -= 1
                img, keypoints_data = load_image_and_predict(idx)
                reset_strawberry_selection()

        elif key == ord('k'):  # skip
            if idx < len(images) - 1:
                idx += 1
                img, keypoints_data = load_image_and_predict(idx)
                reset_strawberry_selection()

        elif key == ord('u'):  # undo last point
            if points:
                points.pop()
                update_display()

        elif key == ord('s'):  # save current screen
            out_dir = Path("evaluation_results") / images[idx].stem
            out_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = out_dir / f"screen_{ts}.png"
            cv2.imwrite(str(out_path), display)
            print(f"Screen saved: {out_path}")

    cv2.destroyAllWindows()
    print(f"Evaluation complete. Results saved to: {csv_path}")

if __name__ == "__main__":
    main() 