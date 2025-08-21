import argparse
import os
from collections import defaultdict
from db_processing.db_manager import DBManager
from db_processing.db_config import init_db
from nas_processing.nas_manager import NASManager
from nas_processing.process_image import upload_data_directory

def group_by_folder(rows):
  folder_map = defaultdict(list)
  for row in rows:
    folder_map[row["folder_path"]].append(row)
  return folder_map

def data_process_main():
  parser = argparse.ArgumentParser(description='NAS + DB Process CLI')
  parser.add_argument('--mode', type=str, required=True, choices=['upload', 'download'])
  parser.add_argument('--file_path', type=str, help="Upload file path")
  parser.add_argument('--upload_folder_path', type=str, help="NAS에 업로드 할 폴더 이름 (경로 이름)")
  parser.add_argument('--file_name', type=str, default=None)
  parser.add_argument('--file_path_db', type=str, default=None, help="Path in database")
  parser.add_argument('--nas_folder_path', type=str, help="NAS에서 조회해올 폴더 경로")
  parser.add_argument('--date_from', type=str, default=None)
  parser.add_argument('--date_to', type=str, default=None)
  parser.add_argument('--file_type', type=str, default=None)
  parser.add_argument('--crop_code', type=str, default=None)
  parser.add_argument('--farm_code', type=str, default=None)
  parser.add_argument('--machine_code', type=str, default=None)
  parser.add_argument('--download_dir', type=str, default=None)
  parser.add_argument('--top_folder_path', type=str, help="NAS에서 불러올 데이터 폴더의 최상위 폴더")
  parser.add_argument('--auto_extract', type=bool, default=False, help="다운로드 폴더 자동 압축 해제 여부")

  args = parser.parse_args()

  init_db()

  if args.mode == 'upload':
    if not args.file_path:
      raise ValueError('--file_path is required for file upload')
    upload_data_directory(args.file_path, base_path=args.upload_folder_path)

  elif args.mode == 'download':
    db = DBManager()
    rows = db.select_file_info(
      file_name=args.file_name,
      file_path=args.file_path_db,
      folder_path=args.nas_folder_path,
      date_from=args.date_from,
      date_to=args.date_to,
      file_type=args.file_type,
      crop_code=args.crop_code,
      farm_code=args.farm_code,
      machine_code=args.machine_code,
      top_folder_path=args.top_folder_path
    )

    print(f"[INFO] total row count: {len(rows)}")

    nas = NASManager()

    if len(rows) == 1:
      print(f"Single file download: {rows[0]['file_name']}")
      local_path = os.path.join(args.download_dir, rows[0]['file_name'])
      nas.download_file(rows[0]['file_path'], local_path=local_path)
    else:
      grouped = group_by_folder(rows=rows)
      for folder_path, group in grouped.items():
        nas.download_folder(group, local_path=args.download_dir, auto_extract=args.auto_extract)

def main():
  data_process_main()

if __name__ == '__main__':
  main()