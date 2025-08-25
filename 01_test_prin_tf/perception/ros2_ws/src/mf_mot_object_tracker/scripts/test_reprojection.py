#!/usr/bin/env python3
"""
Rerun 기반 이미지 T/F 라벨러 (CSV 저장)

기능
- 이미지 폴더와 라벨 폴더(예: YOLO .txt)를 읽어 매 이미지에 대해 True/False 결정을 CSV로 저장
- 기본값은 "라벨 파일 존재 여부"(has_label)로 설정, 수동(--manual)일 때 y/n로 덮어쓰기 가능(Enter=기본값)
- Rerun 뷰어로 이미지를 표시하고(옵션: YOLO 라벨을 오버레이), 터미널에서 입력받아 진행
- 중단 후 재실행하면 기존 CSV를 읽어 이어서 진행(resume)
- undo(이전 이미지 취소), skip, quit 지원

설치
  pip install rerun-sdk opencv-python
사용 예
  python rerun_image_tf_labeler.py \
    --images /path/to/images \
    --labels /path/to/labels \
    --out-csv /path/to/out.csv \
    --manual

메모
- YOLO 라벨(.txt)은 "class cx cy w h"(0~1 정규화) 형식을 가정합니다.
- 라벨 경로는 images 기준 상대경로를 유지한 채 labels 디렉터리에서 동일 구조/이름(.txt)로 찾습니다.
"""
from __future__ import annotations
import argparse
import csv
import os
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import rerun as rr

# -------------- 유틸 --------------

def list_images(root: Path, exts=(".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")) -> List[Path]:
    files: List[Path] = []
    for ext in exts:
        files.extend(root.rglob(f"*{ext}"))
    files.sort()
    return files


def image_to_label_path(img_path: Path, images_root: Path, labels_root: Path, label_ext: str = ".txt") -> Path:
    rel = img_path.relative_to(images_root)
    return (labels_root / rel).with_suffix(label_ext)


def ensure_dir(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)


# YOLO: class cx cy w h (norm)
def read_yolo_txt(label_path: Path) -> List[Tuple[int, float, float, float, float]]:
    items = []
    try:
        with open(label_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                c = int(float(parts[0]))
                cx, cy, w, h = map(float, parts[1:])
                items.append((c, cx, cy, w, h))
    except FileNotFoundError:
        pass
    return items


def draw_yolo_boxes_bgr(img_bgr, yolo_items, class_names=None):
    h, w = img_bgr.shape[:2]
    for it in yolo_items:
        c, cx, cy, ww, hh = it
        x = int((cx - ww / 2) * w)
        y = int((cy - hh / 2) * h)
        x2 = int((cx + ww / 2) * w)
        y2 = int((cy + hh / 2) * h)
        cv2.rectangle(img_bgr, (x, y), (x2, y2), (0, 255, 0), 2)
        label = str(c if class_names is None else class_names[c] if c < len(class_names) else c)
        cv2.putText(img_bgr, label, (x, max(0, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
    return img_bgr


def load_existing(csv_path: Path) -> Dict[str, str]:
    if not csv_path.exists():
        return {}
    decided: Dict[str, str] = {}
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            decided[row["image_path"]] = row.get("decision", "")
    return decided


def append_row(csv_path: Path, row: Dict[str, str], write_header_if_needed=True):
    ensure_dir(csv_path)
    header = ["image_path", "label_path", "has_label", "decision", "note"]
    exists = csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if write_header_if_needed and not exists:
            w.writeheader()
        w.writerow(row)


def rewrite_without_last(csv_path: Path):
    # undo용: 마지막 라인을 제거
    with open(csv_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    if len(lines) <= 1:
        # header만 있거나 빈 경우
        with open(csv_path, "w", encoding="utf-8") as f:
            if lines:
                f.write(lines[0])
        return
    with open(csv_path, "w", encoding="utf-8") as f:
        f.writelines(lines[:-1])


# -------------- 메인 --------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=Path, required=True, help="이미지 루트 폴더")
    ap.add_argument("--labels", type=Path, required=True, help="라벨(.txt 등) 루트 폴더")
    ap.add_argument("--out-csv", type=Path, required=True, help="결과 CSV 경로")
    ap.add_argument("--label-ext", type=str, default=".txt", help="라벨 확장자(기본 .txt)")
    ap.add_argument("--manual", action="store_true", help="수동 y/n 입력으로 결정")
    ap.add_argument("--overlay-labels", action="store_true", help="YOLO 라벨을 이미지 위에 그려서 표시")
    ap.add_argument("--class-names", type=Path, default=None, help="선택: class.names 파일 경로")
    ap.add_argument("--redo", action="store_true", help="기존 CSV가 있어도 처음부터 다시")
    args = ap.parse_args()

    images = list_images(args.images)
    if not images:
        print("[ERR] 이미지가 없습니다.")
        return

    class_names = None
    if args.class_names and args.class_names.exists():
        with open(args.class_names, "r", encoding="utf-8") as f:
            class_names = [ln.strip() for ln in f if ln.strip()]

    decided_map: Dict[str, str] = {} if args.redo else load_existing(args.out_csv)

    rr.init("Rerun Image T/F Labeler")
    # Rerun 뷰어 실행 (한 번만)
    rr.spawn()

    total = len(images)
    idx = 0
    # 재시작 시 이미 결정된 항목은 건너뛰도록 idx 위치 조정
    if decided_map:
        while idx < total and str(images[idx]) in decided_map:
            idx += 1

    print("\n==== 조작법 ====")
    print(" y: True / n: False / Enter: 기본값(라벨 존재 여부) / u: undo / s: skip / q: quit\n")

    while idx < total:
        img_path = images[idx]
        lbl_path = image_to_label_path(img_path, args.images, args.labels, args.label_ext)
        has_label = lbl_path.exists()

        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            print(f"[WARN] 읽기 실패: {img_path}")
            idx += 1
            continue

        show_img = img.copy()
        yolo_items = read_yolo_txt(lbl_path)
        if args.overlay_labels and yolo_items:
            show_img = draw_yolo_boxes_bgr(show_img, yolo_items, class_names)

        # Rerun에 표시 (같은 엔티티 경로에 매 프레임 갱신)
        rr.log("current/image", rr.Image(show_img[:, :, ::-1]))  # BGR->RGB
        rr.log("meta", rr.TextLog(f"{idx+1}/{total} | has_label={has_label} | {img_path.name}"))

        # 터미널 입력
        base_default = "True" if has_label else "False"
        if args.manual:
            ans = input(f"[{idx+1}/{total}] {img_path.name}  기본값={base_default}  (y/n/Enter/u/s/q) > ").strip().lower()
        else:
            ans = ""

        if ans == "q":
            print("[QUIT]")
            break
        if ans == "u":
            # 직전 항목 취소
            rewrite_without_last(args.out_csv)
            # 이전 이미지로 이동(가능하면)
            if idx > 0:
                idx -= 1
            while idx >= 0 and str(images[idx]) not in decided_map:
                # CSV에서 한 줄 뺐으니 decided_map도 동기화 필요: 
                # 안전하게 재로딩
                decided_map = load_existing(args.out_csv)
                break
            continue
        if ans == "s":
            print("[SKIP]")
            idx += 1
            continue

        if ans == "y":
            decision = "True"
        elif ans == "n":
            decision = "False"
        else:
            # Enter 또는 기타 입력 -> 기본값
            decision = base_default

        row = {
            "image_path": str(img_path),
            "label_path": str(lbl_path if has_label else ""),
            "has_label": str(has_label),
            "decision": decision,
            "note": "",
        }
        append_row(args.out_csv, row)
        decided_map[str(img_path)] = decision
        idx += 1

    print("\nDone. CSV =>", args.out_csv)


if __name__ == "__main__":
    main()
