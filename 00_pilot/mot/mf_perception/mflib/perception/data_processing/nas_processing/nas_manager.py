import requests
import os
import base64
import re
import zipfile
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

class NASManager:
  def __init__(self, base_path="/mf_straw"):
    self.domain = os.getenv("NAS_DOMAIN")
    self.port = os.getenv("NAS_PORT")
    self.user = os.getenv("NAS_USER")
    self.password = os.getenv("NAS_PWD")
    self.base_path = base_path
    self.sid = self._get_sid()

  def _get_sid(self):
    """Get NAS login session ID"""
    try:
      # password Encoding
      encoded_pwd = base64.b64encode(self.password.encode('utf-8')).decode('utf-8')
      
      # Get Params
      params = {
        'user': self.user,
        'pwd': encoded_pwd,
        'serviceKey': '1',
        'service': '1',
        'force_to_check_2sc': '0'
      }
      
      # Get 요청
      response = requests.get(
        f"https://{self.domain}:{self.port}/cgi-bin/authLogin.cgi",
        params=params,
        timeout=10
      )
      response.raise_for_status()
      
      if '<authSid>' in response.text:
        sid = response.text.split('<authSid><![CDATA[')[1].split(']]></authSid>')[0]
        return sid
      else:
        print(f"[ERROR] Failed to get sid: {response.text}")
        return None
    except Exception as e:
      print(f"[EXCEPTION] Failed Login: {e}")
      return None
    
  def get_next_seq_folder(self, date_str):
    target_path = f"{self.base_path}/{date_str}"

    requests.post(
      f"https://{self.domain}:{self.port}/cgi-bin/filemanager/utilRequest.cgi",
      params={
        "func":"create_dir",
        "sid": self.sid,
        "dest_folder": date_str,
        "dest_path": self.base_path
      }
    )

    list_resp = requests.get(
      f"https://{self.domain}:{self.port}/cgi-bin/filemanager/utilRequest.cgi",
      params={
        "func": "get_list",
        "sid": self.sid,
        "path": target_path,
        "show_parent": "1",
        "limit": "500"
      }
    )

    seq_folders = []
    if list_resp.ok:
      entries = list_resp.json().get('datas', [])
      for entry in entries:
        name = entry.get('filename', '')
        if re.fullmatch(r'\d{2}', name):
          seq_folders.append(int(name))

    next_seq = max(seq_folders, default=0) + 1
    next_seq_str = f"{next_seq:02d}"

    requests.post(
      f"https://{self.domain}:{self.port}/cgi-bin/filemanager/utilRequest.cgi",
      params = {
        "func": "createdir",
        "sid": self.sid,
        "dest_folder": next_seq_str,
        "dest_path": target_path
      }
    )

    return f"{date_str}/{next_seq_str}"
  
  def create_folder(self, nested_path: str):
    parts = nested_path.strip("/").split("/")
    current_path = self.base_path
    for part in parts:
      resp = requests.post(
        f"https://{self.domain}:{self.port}/cgi-bin/filemanager/utilRequest.cgi",
        params={
          "func": "createdir",
          "sid": self.sid,
          "dest_folder": part,
          "dest_path": current_path
        }
      )
      current_path += f"/{part}"
    return f"{current_path}"

    
  def upload_file(self, file_path, folder=None, max_retries=3,
                  retry_delay=2):
    """파일 업로드 및 폴더 생성 + 재시도 로직"""
    for attempt in range(1, max_retries + 1):
      if not self.sid:
        self.sid = self._get_sid()
        if not self.sid:
          print("[ERROR] No valid session id")
          return False
      
      if not os.path.exists(file_path):
        print(f"[ERROR] No existing file: {file_path}")
        return False
      
      upload_path = f"{self.base_path}/{folder}" if folder else self.base_path
        
      try:
        with open(file_path, 'rb') as f:
          files = {'file': (os.path.basename(file_path), f)}
          upload_resp = requests.post(
            f"https://{self.domain}:{self.port}/cgi-bin/filemanager/utilRequest.cgi",
            params={
              "func": "upload",
              "type": "standard",
              "sid": self.sid,
              "dest_path": upload_path,
              "overwrite": 1,
              "progress": os.path.basename(file_path).replace("/", "-")
            },
            files=files
          )

          print(f"[DEBUG] 파일 업로드 : {upload_resp.text}")
          if '"status": 1' in upload_resp.text and upload_resp.status_code == 200:
            relative_path = f"{upload_path}/{os.path.basename(file_path)}"
            print(F"[INFO] Successfully Uploaded: {relative_path}")
            return relative_path
          else:
            print(f"[WARN] Failed to upload: {upload_resp.text}")
      except Exception as e:
        print(f"[ERROR] Exception during uploading: {e}")

      if attempt < max_retries:
        print(f"[RETRY] Retry uploading after {retry_delay}sec ...")

    print(f"[FAIL] Upload failed. (max {max_retries} times retried): {file_path}")
    return None
  
  def download_file(self, remote_path: str, local_path: str):
    """ Download single file """
    try:
      source_path = os.path.dirname(remote_path)
      source_file = os.path.basename(remote_path)

      resp = requests.get(
        f"https://{self.domain}:{self.port}/cgi-bin/filemanager/utilRequest.cgi",
        params={
          "func": "download",
          "sid": self.sid,
          "isfolder": 0,
          "compress": 0,
          "source_path": source_path,
          "source_file": source_file,
          "source_total": 1
        }
      )

      if resp.status_code == 200:
        with open(local_path, 'wb') as f:
          for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
        return local_path
      else:
        print(f"[ERROR] Download file failed: {resp.status_code} - {resp.text}" )
        return None
      
    except Exception as e:
      print(f"[EXCEPTION] Excpetion occured during download")
      return None
    
  def extract_zip(self, zip_path):
    """ zip file extract for model training """
    extract_to = zip_path.replace(".zip", "")
    os.makedirs(extract_to, exist_ok=True)

    try:
      with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
      print(f"[INFO] Extract completed: {extract_to}")
      return extract_to
    except Exception as e:
      print(f"[ERROR] Failed extracting: {e}")
      return None
  
  def download_files(self, file_rows, download_dir="./downloads", 
                     overwrite=False, is_folder=False):
    """
    Download rows of file
    file_rows = select list data shape ({ '', '', '' })
    download_dir = download directory
    overwrite = If true, overwrites existing file
    """
    """ Deprecated : It will be used someday, but use download_file
    or download_folder instead."""
    os.makedirs(download_dir, exist_ok=True)
    failed = []

    for row in file_rows:
      remote_path = row["file_path"]
      file_name = row["file_name"]
      local_path = os.path.join(download_dir, file_name)

      if not overwrite and os.path.exists(local_path):
        print(f"[SKIP] Existing file: {file_name}")
        continue

      print(f"[DOWNLOADING] {file_name} from NAS to {local_path}.")
      success = self.download_file(remote_path=remote_path, local_path=local_path)

      if not success:
        print(f"[FAIL] Download files failed.")
        failed.append(file_name)
      
    return failed
  
  def download_folder(self, rows, local_path, auto_extract=False):
    """ Downloading folder """
    folder_path = rows[0]["folder_path"]
    zip_name = folder_path.strip("/").replace("/", "_") + ".zip"
    local_zip_path = os.path.join(local_path, zip_name)
    source_path = os.path.dirname(folder_path)
    source_file = os.path.basename(folder_path)

    params = {
      "func": "download",
      "sid": self.sid,
      "isfolder": 1,
      "compress": 1,
      "source_path": source_path,
      "source_file": source_file,
      "source_total": 1
    }
    try:
      resp = requests.get(
        f"https://{self.domain}:{self.port}/cgi-bin/filemanager/utilRequest.cgi",
        params=params,
        stream=True
      )

      if resp.status_code == 200:
        os.makedirs(os.path.dirname(local_zip_path), exist_ok=True)
        with open(local_zip_path, 'wb') as f:
          for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
        print(f"[INFO] Folder download complete: {local_zip_path}")
        if auto_extract:
          extracted_path = self.extract_zip(local_zip_path)
          os.remove(local_zip_path) # 압축 해제 인 경우 삭제
          return extracted_path
        return local_zip_path
      else:
        print(f"[ERROR] Folder download failed: {resp.status_code} - {resp.text}")
        return None
    except Exception as e:
      print(f"[EXCEPTION] Exception occured during folder download: {e}")
      return None