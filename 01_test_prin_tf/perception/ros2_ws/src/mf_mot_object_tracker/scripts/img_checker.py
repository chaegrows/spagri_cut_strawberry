#!/usr/bin/env python3
"""
Rerun 기반 CSV/폴더 T/F 라벨러 (CSV 갱신)

요구사항 반영
- CSV를 **한 줄씩 읽으면서** 특정 경로에 파일이 있으면 Rerun으로 띄우고 터미널에서 **True/False** 입력
- 입력값을 **해당 라인의 마지막 열**에 새 컬럼(기본 헤더: "True-False")으로 추가/갱신하여 **CSV로 저장**
- 재시작 시 이어서 진행(resume), undo/skip/quit 지원
- (옵션) 폴더 스캔 모드도 유지

설치
  pip install rerun-sdk opencv-python

사용 예 (CSV 기반)
  python img_checker.py \
    --mode csv \
    --csv-in ./enhanced_images/tf_data.csv \
    --out-csv tf_data_labeled.csv \
    --path-col image_filename_tf_only \
    --base-dir ./enhanced_images/enhanced_images \
    --manual

* 같은 파일에 덮어쓰고 싶으면 `--inplace` 사용(권장: 백업 자동 생성)

CSV 가정
- CSV 안에 "이미지 파일명 또는 경로"가 들어있는 열이 하나 존재(이름은 `--path-col`로 지정)
- 값이 **상대경로**라면 `--base-dir`와 결합하여 실제 경로를 구성
- 새 컬럼 이름은 기본 "True-False"(하이픈 포함 지원) → `--result-col`로 변경 가능

키 바인딩
- `y` = True, `n` = False, `Enter` = 기본값(기존 값 있으면 그 값, 없으면 False), `u` = 되돌리기, `s` = 건너뛰기, `q` = 종료

참고
- Rerun은 시각화만 담당, 입력은 터미널에서 받습니다.
"""
from __future__ import annotations
import argparse
import csv
import os
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import rerun as rr
import numpy as np
import shutil

# ---------------- 공통 유틸 ----------------
TEXT_COLOR = (0, 0, 0)

def ensure_dir(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)


def draw_info(img_bgr, text: str):
    cv2.putText(img_bgr, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, TEXT_COLOR, 2, cv2.LINE_AA)
    return img_bgr


def show_with_rerun(img_bgr, meta: str):
    rr.log("current/image", rr.Image(img_bgr[:, :, ::-1]))  # BGR->RGB
    rr.log("meta", rr.TextLog(meta))

# --------------- CSV 모드 ---------------

def csv_detect_dialect(csv_path: Path):
    with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
        sample = f.read(2048)
    try:
        sniffer = csv.Sniffer()
        dialect = sniffer.sniff(sample)
        return dialect
    except Exception:
        return csv.excel  # 기본 , 구분자


def read_csv_rows(csv_path: Path) -> Tuple[List[str], List[Dict[str, str]], str]:
    dialect = csv_detect_dialect(csv_path)
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, dialect=dialect)
        rows = list(reader)
        headers = reader.fieldnames or []
    delimiter = getattr(dialect, "delimiter", ",")
    return headers, rows, delimiter


def write_csv_header(fp, headers: List[str], delimiter: str):
    writer = csv.DictWriter(fp, fieldnames=headers, delimiter=delimiter)
    writer.writeheader()
    return writer


def append_csv_row(fp, headers: List[str], row: Dict[str, str], delimiter: str):
    writer = csv.DictWriter(fp, fieldnames=headers, delimiter=delimiter)
    writer.writerow(row)


def truncate_last_line(file_path: Path):
    # 안전한 undo: 텍스트 라인 기준 마지막 데이터 라인 제거
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    if len(lines) <= 1:
        with open(file_path, "w", encoding="utf-8") as f:
            if lines:
                f.write(lines[0])  # header만 유지
        return
    with open(file_path, "w", encoding="utf-8") as f:
        f.writelines(lines[:-1])


def resolve_image_path(val: str, base_dir: Path | None) -> Path:
    p = Path(val)
    if not p.is_absolute() and base_dir is not None:
        p = base_dir / p
    return p


def run_csv_mode(args):
    headers, rows, delimiter = read_csv_rows(args.csv_in)
    if args.result_col not in headers:
        headers_out = [*headers, args.result_col]
    else:
        headers_out = headers

    # 출력 파일 결정
    out_path = args.csv_in if args.inplace else args.out_csv
    if out_path is None:
        raise SystemExit("--out-csv 를 지정하거나 --inplace 를 사용하세요.")

    # 백업 (inplace)
    if args.inplace:
        bak = Path(str(out_path) + ".bak")
        if not bak.exists():
            shutil.copy2(args.csv_in, bak)

    # Rerun 초기화
    rr.init("Rerun CSV T/F Labeler")
    rr.spawn()

    base_dir = Path(args.base_dir) if args.base_dir else None

    # resume: 이미 out CSV가 있으면 몇 줄까지 처리됐는지 파악
    processed = 0
    if Path(out_path).exists():
        with open(out_path, "r", encoding="utf-8") as f:
            processed = max(0, sum(1 for _ in f) - 1)  # header 제외

    # writer 준비 (append 모드)
    ensure_dir(Path(out_path))
    mode = "a" if Path(out_path).exists() and processed > 0 else "w"
    with open(out_path, mode, encoding="utf-8", newline="") as fp:
        if mode == "w":
            write_csv_header(fp, headers_out, delimiter)

        idx = processed
        total = len(rows)

        print("==== 조작법 ====")
        print(" y: True / n: False / Enter: 기본값 / u: undo / s: skip / q: quit")

        while idx < total:
            row = rows[idx]
            img_val = row.get(args.path_col, "").strip()
            img_path = resolve_image_path(img_val, base_dir)

            exists = img_path.exists()
            if not exists:
                decision_default = row.get(args.result_col, "False") or "False"
                if not args.show_even_if_missing:
                    # 표시하지 않고 바로 기록
                    row[args.result_col] = decision_default
                    append_csv_row(fp, headers_out, row, delimiter)
                    print(f"[{idx+1}/{total}] MISSING -> write default={decision_default} : {img_val}")
                    idx += 1
                    continue

            # 이미지 로드 & 표시
            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR) if exists else None
            canvas = img.copy() if img is not None else (255 * np.ones((480, 640, 3), dtype=np.uint8))
            info = f"{idx+1}/{total} | exists={exists} | {img_path.name if img is not None else img_val}"
            canvas = draw_info(canvas, info)
            show_with_rerun(canvas, info)

            # 기본값: 기존 값이 있으면 그 값, 없으면 False
            decision_default = row.get(args.result_col, "False") or "False"
            ans = input(f"[{idx+1}/{total}] {img_val}  기본값={decision_default} (y/n/Enter/u/s/q) > ").strip().lower() if args.manual else ""

            if ans == "q":
                print("[QUIT]")
                break
            if ans == "u":
                truncate_last_line(Path(out_path))
                if idx > 0:
                    idx -= 1
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
                decision = decision_default

            row[args.result_col] = decision
            append_csv_row(fp, headers_out, row, delimiter)
            print(f"[{idx+1}/{total}] -> {decision}")
            idx += 1

    print("Done. CSV =>", out_path)

# --------------- 폴더 모드(기존) ---------------

def list_images(root: Path, exts=(".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")) -> List[Path]:
    files: List[Path] = []
    for ext in exts:
        files.extend(root.rglob(f"*{ext}"))
    files.sort()
    return files


def run_folder_mode(args):
    images = list_images(Path(args.images))
    if not images:
        print("[ERR] 이미지가 없습니다.")
        return

    rr.init("Rerun Folder T/F Labeler")
    rr.spawn()

    import numpy as np

    idx = 0
    total = len(images)

    print("==== 조작법 ====")
    print(" y: True / n: False / Enter=False / u: undo / s: skip / q: quit")

    ensure_dir(Path(args.out_csv))
    header = ["image_path", "decision"]
    exists = Path(args.out_csv).exists()
    with open(args.out_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if not exists:
            w.writeheader()

        while idx < total:
            p = images[idx]
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if img is None:
                idx += 1
                continue
            info = f"{idx+1}/{total} | {p.name}"
            img = draw_info(img, info)
            show_with_rerun(img, info)

            ans = input(f"[{idx+1}/{total}] {p.name} (y/n/Enter/u/s/q) > ").strip().lower() if args.manual else ""
            if ans == "q":
                break
            if ans == "u":
                # 간단히 마지막 라인 삭제
                truncate_last_line(Path(args.out_csv))
                if idx > 0:
                    idx -= 1
                continue
            if ans == "s":
                idx += 1
                continue

            decision = "True" if ans == "y" else ("False" if ans == "n" else "False")
            w.writerow({"image_path": str(p), "decision": decision})
            idx += 1

    print("Done. CSV =>", args.out_csv)

# ---------------- 엔트리포인트 ----------------

def main():
    ap = argparse.ArgumentParser()

    sub = ap.add_subparsers(dest="mode", required=False)

    # CSV 모드
    ap.add_argument("--mode", choices=["csv", "folder"], default="csv", help="동작 모드")

    # 공통 옵션
    ap.add_argument("--manual", action="store_true", help="터미널에서 y/n 입력 받기")

    # CSV 모드 옵션
    ap.add_argument("--csv-in", type=Path, help="입력 CSV 경로")
    ap.add_argument("--out-csv", type=Path, help="출력 CSV 경로(미지정시 --inplace 필요)")
    ap.add_argument("--inplace", action="store_true", help="입력 CSV에 컬럼 추가하여 제자리 갱신(백업 .bak 생성)")
    ap.add_argument("--path-col", type=str, default="image_path", help="이미지 경로/파일명이 들어있는 열 이름")
    ap.add_argument("--base-dir", type=str, default=None, help="상대경로일 때 앞에 붙일 루트 경로")
    ap.add_argument("--result-col", type=str, default="True-False", help="추가/갱신할 결과 컬럼명")
    ap.add_argument("--show-even-if-missing", action="store_true", help="파일이 없어도 뷰어에 표시(경고 캔버스)")

    # 폴더 모드 옵션(레거시)
    ap.add_argument("--images", type=Path, help="폴더 모드: 이미지 루트")

    args = ap.parse_args()

    if args.mode == "csv":
        if not args.csv_in:
            raise SystemExit("--csv-in 을 지정하세요.")
        run_csv_mode(args)
    else:
        if not args.images or not args.out_csv:
            raise SystemExit("폴더 모드에는 --images 와 --out-csv 가 필요합니다.")
        run_folder_mode(args)


if __name__ == "__main__":
    main()

