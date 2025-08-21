#!/bin/bash

FOLDER_ID="1Ly_eElD_ywGnZnPpuSxlpiSpECmaHHMf"
OUTPUT_DIR="ai_models"

echo "[Downloads] Google Drive pcd...."
gdown --folder "https://drive.google.com/drive/folders/${FOLDER_ID}" -O "${OUTPUT_DIR}"

cd "${OUTPUT_DIR}" || { echo "❌ 폴더 이동 실패"; exit 1; }

echo "✅ DONE! check pt folder!"


