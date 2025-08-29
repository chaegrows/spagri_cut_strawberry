#!/usr/bin/env python3
"""
OpenCV 기반 CSV 이미지 T/F 라벨러 (Rerun 불필요)

기능
- CSV를 한 줄씩 읽어서 특정 컬럼(--path-col)에서 이미지 경로/파일명을 가져옴
- (상대경로면 --base-dir와 결합) 이미지를 OpenCV 창으로 표시하고 키 입력으로 True/False 결정
- 결정값을 CSV의 마지막 열(--result-col, 기본 "True-False")에 기록
- 중단 후 재실행하면 기존 out CSV를 읽어 이어서 진행(resume)
- undo/skip/quit 지원

설치
  pip install opencv-python

예시
  python 004_keypoints_test.py \
    --csv-in ./enhanced_images/tf_data.csv \
    --out-csv ./enhanced_images/tf_data_labeled_cv2.csv \
    --path-col image_filename_tf_only \
    --base-dir ./enhanced_images/enhanced_images \
    --scale 0.9 \
    --manual

키 바인딩
  y: True / n: False / Enter: 기본값(기존 값/없으면 False)
  u: undo(마지막 기록 취소) / s: skip / q: quit
  [창은 드래그/리사이즈 가능]
"""
from __future__ import annotations
import argparse
import csv
import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np


# ---------- 유틸 ----------
def ensure_dir(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)


def csv_detect_dialect(csv_path: Path):
    with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
        sample = f.read(4096)
    try:
        sniffer = csv.Sniffer()
        d = sniffer.sniff(sample)
        return d
    except Exception:
        return csv.excel  # 기본 ,


def read_csv_rows(csv_path: Path) -> Tuple[List[str], List[Dict[str, str]], str]:
    dialect = csv_detect_dialect(csv_path)
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, dialect=dialect)
        rows = list(reader)
        headers = reader.fieldnames or []
    delimiter = getattr(dialect, "delimiter", ",")
    return headers, rows, delimiter


def write_csv_header(fp, headers: List[str], delimiter: str):
    w = csv.DictWriter(fp, fieldnames=headers, delimiter=delimiter)
    w.writeheader()
    return w


def append_csv_row(fp, headers: List[str], row: Dict[str, str], delimiter: str):
    w = csv.DictWriter(fp, fieldnames=headers, delimiter=delimiter)
    w.writerow(row)


def truncate_last_line(file_path: Path):
    # 텍스트 라인 기준 마지막 데이터 라인 제거(undo)
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    if len(lines) <= 1:
        with open(file_path, "w", encoding="utf-8") as f:
            if lines:
                f.write(lines[0])
        return
    with open(file_path, "w", encoding="utf-8") as f:
        f.writelines(lines[:-1])


def resolve_image_path(val: str, base_dir: Path | None) -> Path:
    p = Path(val)
    if not p.is_absolute() and base_dir is not None:
        p = base_dir / p
    return p


def put_multi_line_text(img, lines, org=(10, 30), line_height=26, color=(255,255,255), thickness=2, scale=0.7):
    x, y = org
    for i, line in enumerate(lines):
        cv2.putText(img, line, (x, y + i*line_height), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_bbox(img, row, scale=1.0):
    """Draw bounding box on image if bbox coordinates exist in the row"""
    try:
        bbox_x1 = float(row.get('bbox_x1', 0) or 0) * scale
        bbox_y1 = float(row.get('bbox_y1', 0) or 0) * scale
        bbox_x2 = float(row.get('bbox_x2', 0) or 0) * scale
        bbox_y2 = float(row.get('bbox_y2', 0) or 0) * scale
        
        # Check if bbox coordinates are valid
        if bbox_x1 > 0 and bbox_y1 > 0 and bbox_x2 > bbox_x1 and bbox_y2 > bbox_y1:
            # Draw bounding box rectangle
            cv2.rectangle(img, (int(bbox_x1), int(bbox_y1)), (int(bbox_x2), int(bbox_y2)), 
                         (0, 255, 0), 2)  # Green color, thickness 2
            
            # Add bbox label if exists
            bbox_label = row.get('bbox_label', '')
            if bbox_label:
                label_text = f"{bbox_label}"
                label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                # Draw label background
                cv2.rectangle(img, (int(bbox_x1), int(bbox_y1) - label_size[1] - 10),
                             (int(bbox_x1) + label_size[0] + 10, int(bbox_y1)), (0, 255, 0), -1)
                # Draw label text
                cv2.putText(img, label_text, (int(bbox_x1) + 5, int(bbox_y1) - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            # Add bbox coordinates info (original coordinates)
            orig_x1 = int(float(row.get('bbox_x1', 0) or 0))
            orig_y1 = int(float(row.get('bbox_y1', 0) or 0))
            orig_x2 = int(float(row.get('bbox_x2', 0) or 0))
            orig_y2 = int(float(row.get('bbox_y2', 0) or 0))
            bbox_info = f"bbox: ({orig_x1},{orig_y1}) - ({orig_x2},{orig_y2})"
            return bbox_info
    except (ValueError, TypeError):
        pass
    
    return None


def draw_keypoints(img, row, scale=1.0):
    """Draw keypoints on image if keypoint coordinates exist in the row"""
    keypoints_info = []
    colors = [(255, 0, 0), (0, 0, 255), (255, 255, 0)]  # Blue, Red, Yellow for keypoints 0, 1, 2
    
    try:
        for i in range(3):  # Assuming 3 keypoints (0, 1, 2)
            kp_x_key = f'keypoint_{i}_x'
            kp_y_key = f'keypoint_{i}_y'
            kp_conf_key = f'keypoint_{i}_conf'
            
            kp_x = row.get(kp_x_key, '')
            kp_y = row.get(kp_y_key, '')
            kp_conf = row.get(kp_conf_key, '')
            
            if kp_x and kp_y and kp_conf:
                try:
                    x = float(kp_x) * scale
                    y = float(kp_y) * scale
                    conf = float(kp_conf)
                    
                    if conf > 0.5:  # Only draw keypoints with confidence > 0.5
                        # Draw keypoint as circle
                        # cv2.circle(img, (int(x), int(y)), 5, colors[i], -1)
                        # Draw keypoint number
                        # cv2.putText(img, str(i), (int(x) + 8, int(y) - 8),
                        #            cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[i], 2)
                        
                        # Use original coordinates for info display
                        orig_x = int(float(kp_x))
                        orig_y = int(float(kp_y))
                        keypoints_info.append(f"kp{i}: ({orig_x},{orig_y}) conf:{conf:.3f}")
                except (ValueError, TypeError):
                    continue
    except Exception:
        pass
    
    return keypoints_info


# ---------- 메인 로직 ----------
def run(args):
    headers, rows, delimiter = read_csv_rows(args.csv_in)
    if args.path_col not in headers:
        raise SystemExit(f"--path-col '{args.path_col}' 이(가) CSV 헤더에 없습니다. 현재 헤더: {headers}")
    if args.result_col not in headers:
        headers_out = [*headers, args.result_col]
    else:
        headers_out = headers

    out_path: Path = args.csv_in if args.inplace else (args.out_csv or Path(args.csv_in).with_suffix(".labeled.csv"))
    if out_path is None:
        raise SystemExit("--out-csv 를 지정하거나 --inplace 를 사용하세요.")

    # inplace 백업
    if args.inplace:
        bak = Path(str(out_path) + ".bak")
        if not bak.exists():
            shutil.copy2(args.csv_in, bak)

    # resume: out CSV가 있으면 처리된 라인 수 파악
    processed = 0
    if out_path.exists():
        with open(out_path, "r", encoding="utf-8") as f:
            processed = max(0, sum(1 for _ in f) - 1)

    ensure_dir(out_path)
    mode = "a" if out_path.exists() and processed > 0 else "w"
    fp = open(out_path, mode, encoding="utf-8", newline="")
    try:
        if mode == "w":
            write_csv_header(fp, headers_out, delimiter)

        idx = processed
        total = len(rows)

        cv2.namedWindow("TF Labeler (cv2)", cv2.WINDOW_FULLSCREEN)
        # cv2.namedWindow("TF Labeler (cv2)", cv2.WINDOW_NORMAL)
        # cv2.setWindowProperty("TF Labeler (cv2)", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
        print("\n==== 조작법 ====")
        print(" y: True / n: False / Enter: 기본값 / u: undo / s: skip / q: quit\n")

        while idx < total:
            row = rows[idx]
            img_val = (row.get(args.path_col, "") or "").strip()
            img_path = resolve_image_path(img_val, Path(args.base_dir) if args.base_dir else None)
            exists = img_path.exists()

            # 이미지 로드/스케일
            if exists:
                img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            else:
                img = None

            if img is None:
                canvas = np.full((480, 800, 3), 20, dtype=np.uint8)
                put_multi_line_text(canvas, [
                    f"[{idx+1}/{total}] (파일 없음)",
                    f"path-col: {args.path_col}",
                    f"value: {img_val}",
                    "Press: y/n/Enter/u/s/q"
                ], org=(10, 40), color=(0, 0, 255))
            else:
                if args.scale and args.scale != 1.0:
                    h, w = img.shape[:2]
                    img = cv2.resize(img, (int(w*args.scale), int(h*args.scale)))
                canvas = img.copy()
                
                # Draw bounding box and keypoints with scale factor
                current_scale = args.scale if args.scale else 1.0
                bbox_info = draw_bbox(canvas, row, current_scale)
                keypoints_info = draw_keypoints(canvas, row, current_scale)
                
                info_lines = [
                    f"[{idx+1}/{total}] exists=True",
                    f"{img_path.name}",
                    f"default={row.get(args.result_col, 'False') or 'False'}",
                ]
                
                # Add bbox info if available
                if bbox_info:
                    info_lines.append(bbox_info)
                
                # Add keypoints info if available
                if keypoints_info:
                    info_lines.extend(keypoints_info)
                
                info_lines.append("Press: y/n/Enter/u/s/q")
                
                put_multi_line_text(canvas, info_lines)

            cv2.imshow("TF Labeler (cv2)", canvas)
            key = cv2.waitKey(0) & 0xFF  # 블로킹

            # 해석
            if key in (ord('q'), ord('Q')):
                print("[QUIT]")
                break
            if key in (ord('u'), ord('U')):
                truncate_last_line(out_path)
                if idx > 0:
                    idx -= 1
                continue
            if key in (ord('s'), ord('S')):
                print("[SKIP]")
                idx += 1
                continue

            if key in (ord('y'), ord('Y')):
                decision = "True"
            elif key in (ord('n'), ord('N')):
                decision = "False"
            elif key in (13, 10):  # Enter
                decision = row.get(args.result_col, "False") or "False"
            else:
                # 기타 키는 무시하고 계속 대기
                continue

            row[args.result_col] = decision
            append_csv_row(fp, headers_out, row, delimiter)
            fp.flush()
            os.fsync(fp.fileno())
            print(f"[{idx+1}/{total}] -> {decision}")
            idx += 1

    finally:
        fp.close()
        cv2.destroyAllWindows()

    print("\nDone. CSV =>", out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-in", type=Path, required=True, help="입력 CSV 경로")
    ap.add_argument("--out-csv", type=Path, help="출력 CSV 경로 (미지정시 .labeled.csv)")
    ap.add_argument("--inplace", action="store_true", help="원본 CSV에 결과 컬럼을 추가하여 저장(자동 백업 .bak 생성)")
    ap.add_argument("--path-col", type=str, required=True, help="이미지 경로/파일명이 들어있는 열 이름")
    ap.add_argument("--base-dir", type=str, help="상대경로일 때 앞에 붙일 루트 경로")
    ap.add_argument("--result-col", type=str, default="True-False", help="추가/갱신할 결과 컬럼명")
    ap.add_argument("--scale", type=float, default=1.0, help="cv2 표시 스케일(예: 0.75)")
    ap.add_argument("--manual", action="store_true", help="(호환용 플래그, 동작에는 영향 없음)")

    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
