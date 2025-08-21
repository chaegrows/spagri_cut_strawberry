# 📦 NAS + DB 파일 업로드 및 다운로드 사용법

## 📁 예시 폴더 구조

```
~ (Projectroot)
├── mf_perception
  └── mflib/
    └── perception/
        └──data_processing/
          └── db_processing/
            ├── create_table.py
            ├── db_config.py
            └── etc ...
          └── nas_processing/
            ├── nas_manager.py
            └── process_image.py
          └── data_process_main.py # < 실행 스크립트 >
          └── run_data_process.sh # < 실행 bash >
  └── data/
    └── img_data/
        └── 01/
            └── A/
                ├── image1.jpg
                ├── image2.jpg
            └── B/
                ├── image1.jpg
                ├── image2.jpg
```

---

## 🧭 사용법

### ✅ 업로드

```bash
python ~/data_process_main.py \
  --mode upload \
  --file_path 업로드할 파일 경로 \
  --upload_folder_path nas에 업로드할 경로 (/nas_folder_name)
```

📌 업로드 결과:

```
NAS 경로: /nas 지정 폴더/업로드할 파일 경로/업로드날짜(yyyy-mm-dd)/Sequence(01, 02 ...)/내부 폴더/파일명
```

---

### ✅ 다운로드 (단일 파일)

```bash
python cli/data_process_main.py \
  --mode download \
  --file_name 파일명 \
  --download_dir ./downloads
```

→ 지정한 파일 하나만 NAS에서 다운로드합니다.

---

### ✅ 다운로드 (폴더 압축 다운로드 + 자동 해제)

```bash
python cli/data_process_main.py \
  --mode download \
  --nas_folder_path /camera_data/test_folder/20250530/A \ (또는 top_folder_path로 지정하여 최상위 디렉토리 명을 줘도 됨 ex : test_folder)
  --download_dir ./downloads
```

→ 해당 NAS 폴더 단위로 zip 압축 파일로 다운로드받고 자동으로 압축 해제됩니다.

---

## ⚙️ 주요 옵션 설명 (입력 데이터 /projectroot/data/test_folder/A/test1.png , NAS 경로 /test/test_folder/yyyymmdd/seq/A/test1.png 기준)

| 옵션 | 설명 | 추가 설명 |
|------|------|------|
| `--mode` | `upload` 또는 `download` 선택 | |
| `--file_path` | 업로드할 로컬 폴더의 루트 경로 | |
| `--upload_folder_path` | NAS에서 기준이 되는 상위 경로 | NAS 경로 기준 /test |
| `--download_dir` | NAS에서 받은 파일/폴더를 저장할 로컬 경로 | 원하는 경로 지정 ex : ./downloaded|
| `--file_name` | 다운로드할 파일 이름 (정확히 일치) | file명은 test1.png |
| `--file_path_db` | DB 상의 정확한 파일 경로 | file_path_db는 기준 전체를 다 넣으면 됩니다 |
| `--nas_folder_path` | NAS 상의 특정 폴더 기준으로 다운로드 | nas_folder_path는 /test/test_folder/yyyymmdd/seq/A |
| `--top_folder_path` | DB 검색 시 NAS 기준 최상위 경로 지정 | top_folder_path는 /test_folder로 들어가게 됨. 최종적으로 저장하려고 하는 폴더의 명이 top_folder_path |
| `--date_from`, `--date_to` | 파일 등록일 필터 (YYYY-MM-DD) | |
| `--file_type`, `--crop_code`, `--farm_code`, `--machine_code` | 다양한 검색 필터 조건 | |

---

## 📌 비고

- 업로드 시 폴더 구조는 그대로 NAS에 반영되며, DB에도 메타데이터가 저장됩니다.
- 다운로드는 단일 파일 또는 폴더 압축 방식으로 제공됩니다.
- 내부적으로 NAS는 QNAP API를 기반으로 연동됩니다.

---

## 🔗 예제 결과 경로

예를 들어 다음과 같은 경로가 있고 :
~ (Projectroot)
├── mf_perception
  └── mflib/
    └── perception/
        └──data_processing/
          └── db_processing/
            ├── create_table.py
            ├── db_config.py
            └── etc ...
          └── nas_processing/
  └── data/
    └── A
      └── image1.jpg

아래와 같은 입력이 들어오면 : 

```bash
--file_path ~/mf_perception/data \
--upload_folder_path /camera_data
```

최종 NAS 경로는 다음과 같이 생성됩니다:

```
/camera_data/data/20250530/01/A/image1.jpg
```
