#!/bin/bash

PROJECT_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODE="" # upload | download
DOWNLOAD_DIR="$PROJECT_ROOT/downloads"

FILE_PATH=""
FILE_NAME=""
FOLDER_PATH=""
FILE_PATH_DB=""
FILE_TYPE=""
DATE_FROM=""
DATE_TO=""
CROP_CODE=""
FARM_CODE=""
MACHINE_CODE=""

function usage() {
  echo ""
  echo "Instruction:"
  echo " ./run_data_process.sh --mode [upload|download] [other options]"
  echo ""
  echo "Upload arguments"
  echo " --file_path [path for upload folder]"
  echo "Download arguments"
  echo " --file_name, --folder_path, --crop_code, --farm_code etc.."
  echo " --download_dir [local download directory]"
  echo ""
  exit 1
}

while [[ $# -gt 0 ]]; do
  case $1 in
    --mode) MODE="$2"; shift ;;
    --file_path) FILE_PATH="$2"; shift;;
    --file_name) FILE_NAME="$2"; shift;;
    --file_type) FILE_TYPE="$2"; shift;;
    --folder_path) FOLDER_PATH="$2"; shift;;
    --file_path_db) FILE_PATH_DB="$2"; shift;;
    --date_from) DATE_FROM="$2"; shift;;
    --date_to) DATE_TO="$2"; shift;;
    --crop_code) CROP_CODE="$2"; shift;;
    --farm_code) FARM_CODE="$2"; shift;;
    --machine_code) MACHINE_CODE="$2"; shift;;
    --download_dir) DOWNLOAD_DIR="$2"; shift;;
  esac
  shift
done

echo "[DEBUG] Received MODE: $MODE"

if [[ "$MODE" != "upload" && "$MODE" != "download" ]]; then
  echo "--mode should be upload or download"
  usage
fi

# activate virtual environment mode
source "$PROJECT_ROOT/.venv/bin/activate"

# Upload
if [[ "$MODE" == "upload" ]]; then
  if [[ -z "$FILE_PATH" ]]; then
    echo "--file_path is required for upload"
    exit 1
  fi

  echo "Uploading from: $FILE_PATH ..."
  python "$SCRIPT_DIR/data_process_main.py" --mode upload --file_path "$FILE_PATH"
fi

# Download
if [[ "$MODE" == "download" ]]; then
  echo "[DEBUG] running: python $SCRIPT_DIR/data_process_main.py --mode download ..."
  echo "Downloading based on given filters..."
  python "$SCRIPT_DIR/data_process_main.py" \
    --mode "$MODE" \
    --file_name "$FILE_NAME" \
    --file_path_db "$FILE_PATH_DB" \
    --folder_path "$FOLDER_PATH" \
    --date_from "$DATE_FROM" \
    --date_to "$DATE_TO" \
    --file_type "$FILE_TYPE" \
    --crop_code "$CROP_CODE" \
    --farm_code "$FARM_CODE" \
    --machine_code "$MACHINE_CODE" \
    --download_dir "$DOWNLOAD_DIR"
  EXIT_CODE=$?
  echo "[DEBUG] python exited with code $EXIT_CODE"
fi
echo "[$MODE] Completed!"