import os
import argparse
from pathlib import Path
from datetime import datetime
from nas_processing.nas_manager import NASManager
from db_processing.db_manager import DBManager
from db_processing.db_config import init_db

def process_single_file(abs_path, uploader, inserter, folder=None, description="",
                        farm_code='mf_isu_001', crop_code='C001', machine_code='M001',
                        top_folder_path=None):
  relative_path = uploader.upload_file(abs_path, folder=folder)
  if not relative_path:
    print(f"[ERROR] Upload failed: {abs_path}")
    return
  
  file_size = os.path.getsize(abs_path)
  ext = Path(abs_path).suffix.lower()
  file_type = (
    "image" if ext in ['.png', '.jpg', '.jpeg']
    else "lidar" if ext in ['.npz', '.pcd']
    else "metadata"
  )
  captured_at = datetime.now().date()

  folder_path = os.path.dirname(relative_path)

  inserter.insert_file_info(
    file_path=relative_path,
    farm_code=farm_code,
    crop_code=crop_code,
    machine_code=machine_code,
    file_type=file_type,
    file_name=os.path.basename(abs_path),
    file_size=file_size,
    description=description,
    shoot_date=captured_at,
    folder_path=folder_path,
    top_folder_path=top_folder_path
  )

def single_image_upload(file_path, farm_code='mf_isu_001', crop_code='C001',
                        machine_code='M001'):
  init_db()
  uploader = NASManager()
  inserter = DBManager()
  process_single_file(str(file_path), uploader, inserter,
                      folder=None, farm_code=farm_code, crop_code=crop_code,
                      machine_code=machine_code)

# uploading all data inside folder
def upload_data_directory(data_root_path, farm_code='mf_isu_001', crop_code='C001',
                          machine_code='M001', base_path = "/camera_data"):
  init_db()
  uploader = NASManager(base_path=base_path)
  inserter = DBManager()

  data_root = Path(data_root_path).expanduser().resolve()
  root_folder_name = data_root.name
  date_str = datetime.now().strftime('%Y%m%d')
  seq_folder = uploader.get_next_seq_folder(date_str)

  dirs_to_create = []
  files_to_upload = []

  for root, dirs, files in os.walk(data_root_path):
    rel_dir = os.path.relpath(root, data_root_path)
    if rel_dir == ".":
        rel_dir = ""
    target_folder = f"{root_folder_name}/{seq_folder}/{rel_dir}".replace("\\", "/")
    dirs_to_create.append(target_folder)

    for file in files:
        file_path = Path(root) / file
        target_path = f"{target_folder}/{file}".replace("\\", "/")
        files_to_upload.append((file_path, os.path.dirname(target_path)))


  for folder_path in sorted(dirs_to_create, key=lambda x: str(x).count('/')):
    uploader.create_folder(str(folder_path))

  for file_path, folder in files_to_upload:
    process_single_file(str(file_path), uploader, inserter,
                        folder=folder,
                        farm_code=farm_code, crop_code=crop_code,
                        machine_code=machine_code,
                        top_folder_path=f"/{str(data_root)}")

  return str(data_root)
      
def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--file_path', type=str, required=True, help="Requires file path to upload NAS and database.")

  args = parser.parse_args()

  upload_data_directory(args.file_path)

if __name__ == '__main__':
  main()
  
  # root_dir = Path(__file__).resolve().parents[3]
  # my_file_path = root_dir / "data" / "db" / "source" / "sensors" / "sj_straw_room1_mot_allmap" / "femto_bolt_color_image_raw_compressed" / "frame000000.png"
  # processing_image_upload(
  #   file_path=my_file_path
  # )
  # my_folder_path = root_dir / "data" / "db" / "source" / "sensors" / "sj_straw_room1_mot_allmap"
  # my_folder_path = root_dir / "test_data_2"
  # upload_data_directory(my_folder_path)

  # 조회 및 다운로드 로직
  # init_db()

  # searcher = DBManager()

  # rows = searcher.select_file_info(folder_path="/camera_data/20250508/02/femto_bolt_depth_image_raw_compressedDepth")

  # downloader = NASManager()
  # downloader.download_files(rows, download_dir= root_dir / "test_download")
  # downloader.download_folder(rows, local_path=root_dir / "test_folder", auto_extract=True)