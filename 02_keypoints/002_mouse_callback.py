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

"""
[Batch Angle-from-(-Y) Tool]
- 폴더 내 이미지를 순회하며, 각 이미지에서 다중 쌍(P1->P2) 각도를
  이미지 -Y축(위쪽) 기준 시계방향으로 계산해 CSV에 누적 저장합니다.
- angle_deg = (atan2(vx, -vy) * 180/pi) % 360
  위쪽: 0°, 오른쪽: 90°, 아래쪽: 180°, 왼쪽: 270°

조작:
  - 좌클릭: 점 찍기(P1->P2). 두 점이 모이면 각도 계산 & 미리보기
  - c: 현재 측정값 확정 & CSV 저장
  - r: 현재 측정 취소 & 다시 시작
  - u: 마지막 점(한 점) 되돌리기
  - s: 현재 화면(주석 포함) 저장
  - n/Enter: 다음 이미지
  - p: 이전 이미지
  - k: 건너뛰기(다음)
  - q/ESC: 종료
"""

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
CSV_HEADER = [
    "timestamp",
    "image",
    "pair_idx",
    "p1_x", "p1_y",
    "p2_x", "p2_y",
    "angle_cw_from_negY_deg",
    "angle_unsigned_deg",
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
    ap.add_argument("--out", default="angles.csv", help="Output CSV path")
    args = ap.parse_args()

    folder = Path(args.folder)
    if not folder.exists() or not folder.is_dir():
        raise NotADirectoryError(f"Folder does not exist: {folder}")

    images = list_images(folder)
    if not images:
        raise FileNotFoundError(f"Image not found: {folder} ({IMG_EXTS})")

    csv_path = Path(args.out).resolve()
    ensure_csv_with_header(csv_path)

    window = "Batch"
    cv2.namedWindow(window, cv2.WINDOW_FULLSCREEN)

    idx = 0
    points = []          # 현재 이미지에서 임시 점(최대 2)
    pair_idx = 0         # 현재 이미지에서 기록된 쌍의 인덱스
    display = None
    last_overlay = None  # 마지막 주석 화면(저장용)
    pending_measurement = None  # 저장 대기 중인 측정값

    def load_image(i):
        img = cv2.imread(str(images[i]))
        if img is None:
            raise FileNotFoundError(f"Cannot open image: {images[i]}")
        return img

    img = load_image(idx)
    display = img.copy()
    draw_hud(display, [
        f"[{idx+1}/{len(images)}] {images[idx].name}",
        "Left click: pick P1->P2; c: confirm & save; r: reset",
        "u: undo point; s: save screen",
        "n/Enter: next, p: prev, k: skip, q/ESC: quit",
        "Angle = clockwise from image -Y (upwards)",
        f"Pairs on this image: {pair_idx}"
    ])
    cv2.imshow(window, display)
    cv2.waitKey(150)

    def on_mouse(event, x, y, flags, param):
        nonlocal points, display, last_overlay, pending_measurement, img
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(points) < 2:
                points.append((x, y))

            if len(points) == 2:
                p1, p2 = points
                overlay = img.copy()

                # 사각 표식/라인
                cv2.circle(overlay, p1, 5, (0, 255, 0), -1, cv2.LINE_AA)
                cv2.circle(overlay, p2, 5, (0, 0, 255), -1, cv2.LINE_AA)
                cv2.line(overlay, p1, p2, (255, 0, 0), 2, cv2.LINE_AA)

                # 기준(-Y) 화살표
                draw_reference_arrow(overlay, p1, length=60)

                # 각도 계산
                angle_cw, angle_unsigned = compute_angles(p1, p2)

                if angle_cw is None:
                    msg = ["Select two different points"]
                    pending_measurement = None
                else:
                    # 측정값을 pending에 저장 (아직 CSV에 저장하지 않음)
                    ts = datetime.now().isoformat(timespec="seconds")
                    pending_measurement = {
                        'ts': ts,
                        'p1': p1,
                        'p2': p2,
                        'angle_cw': angle_cw,
                        'angle_unsigned': angle_unsigned,
                        'overlay': overlay
                    }
                    
                    # 주석 텍스트
                    txt = f"{angle_cw:.2f}deg cw from -Y"
                    txt_org = (p1[0] + 10, p1[1] - 10)
                    cv2.putText(overlay, txt, txt_org,
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
                    cv2.putText(overlay, txt, txt_org,
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

                    msg = [
                        f"Angle: {angle_cw:.2f}deg (Press 'c' to save, 'r' to reset)"
                    ]

                # HUD 갱신
                hud = [
                    f"[{idx+1}/{len(images)}] {images[idx].name}",
                    "Left click: pick P1->P2; c: confirm & save; r: reset",
                    "u: undo point; s: save screen",
                    "n/Enter: next, p: prev, k: skip, q/ESC: quit",
                    f"Pairs on this image: {pair_idx}",
                ] + msg
                draw_hud(overlay, hud, org=(10, 26))

                display = overlay
                points.clear()  # 다음 쌍 입력을 위해 초기화

    cv2.setMouseCallback(window, on_mouse)

    def refresh_screen():
        """현재 이미지/상태로 HUD 다시 그림."""
        nonlocal display, img, idx, pair_idx, pending_measurement
        display = img.copy()
        status_msg = ""
        if pending_measurement:
            status_msg = " (Measurement pending - press 'c' to save)"
        draw_hud(display, [
            f"[{idx+1}/{len(images)}] {images[idx].name}",
            "Left click: pick P1->P2; c: confirm & save; r: reset",
            "u: undo point; s: save screen",
            "n/Enter: next, p: prev, k: skip, q/ESC: quit",
            f"Pairs on this image: {pair_idx}{status_msg}"
        ], org=(10, 26))
        cv2.imshow(window, display)

    def save_pending_measurement():
        """대기 중인 측정값을 CSV에 저장"""
        nonlocal pending_measurement, pair_idx, display, last_overlay
        if pending_measurement is None:
            return
        
        # CSV 저장
        row = [
            pending_measurement['ts'],
            images[idx].name,
            pair_idx,
            pending_measurement['p1'][0], pending_measurement['p1'][1],
            pending_measurement['p2'][0], pending_measurement['p2'][1],
            f"{pending_measurement['angle_cw']:.6f}",
            f"{pending_measurement['angle_unsigned']:.6f}",
        ]
        append_csv_row(csv_path, row)

        # 주석 이미지 저장
        out_dir = Path("angle_results") / images[idx].stem
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"pair_{pair_idx + 1:03d}.png"
        cv2.imwrite(str(out_path), pending_measurement['overlay'])

        # 상태 업데이트
        pair_idx += 1
        last_overlay = pending_measurement['overlay'].copy()
        
        # 성공 메시지 표시
        success_overlay = pending_measurement['overlay'].copy()
        success_hud = [
            f"[{idx+1}/{len(images)}] {images[idx].name}",
            "Left click: pick P1->P2; c: confirm & save; r: reset",
            "u: undo point; s: save screen",
            "n/Enter: next, p: prev, k: skip, q/ESC: quit",
            f"Pairs on this image: {pair_idx}",
            f"Saved row -> {csv_path.name}",
            f"Annotated -> {out_path}"
        ]
        draw_hud(success_overlay, success_hud, org=(10, 26))
        display = success_overlay
        
        pending_measurement = None

    def reset_measurement():
        """현재 측정을 취소하고 초기 상태로 돌아감"""
        nonlocal pending_measurement, points
        pending_measurement = None
        points.clear()
        refresh_screen()

    while True:
        cv2.imshow(window, display)
        key = cv2.waitKey(20) & 0xFF

        if key in (27, ord('q')):  # ESC / q
            break

        elif key == ord('c'):  # confirm and save
            save_pending_measurement()

        elif key == ord('r'):  # reset current measurement
            reset_measurement()

        elif key in (ord('\r'), ord('\n'), ord('n')):  # Enter / n
            if idx < len(images) - 1:
                idx += 1
                img = load_image(idx)
                pair_idx = 0
                points.clear()
                pending_measurement = None
                refresh_screen()

        elif key == ord('p'):
            if idx > 0:
                idx -= 1
                img = load_image(idx)
                pair_idx = 0
                points.clear()
                pending_measurement = None
                refresh_screen()

        elif key == ord('k'):  # skip -> next
            if idx < len(images) - 1:
                idx += 1
                img = load_image(idx)
                pair_idx = 0
                points.clear()
                pending_measurement = None
                refresh_screen()

        elif key == ord('u'):  # undo last point
            if points:
                points.pop()
                refresh_screen()

        elif key == ord('s'):  # save current screen
            out_dir = Path("angle_results") / images[idx].stem
            out_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = out_dir / f"screen_{ts}.png"
            cv2.imwrite(str(out_path), display)
            disp2 = display.copy()
            draw_hud(disp2, [f"Screen saved: {out_path}"], org=(10, display.shape[0]-15))
            display = disp2

    cv2.destroyAllWindows()
    print(f"Complete. Result CSV: {csv_path}")

if __name__ == "__main__":
    main()
