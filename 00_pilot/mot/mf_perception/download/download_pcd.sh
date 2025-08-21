#!/bin/bash
# pip install gdown
FOLDER_ID="10ecduyOh_xu8TdY3rT-C2bArGpK355rG"
KEYWORD=${1:-"isu_sj_straw"}  # 인자로 받되, 없으면 기본값 사용
OUTPUT_DIR="pcd"

echo "[Downloads] Google Drive pcd...."
gdown --folder "https://drive.google.com/drive/folders/${FOLDER_ID}" -O "${OUTPUT_DIR}"

cd "${OUTPUT_DIR}" || { echo "❌ 폴더 이동 실패"; exit 1; }
# === 키워드를 포함하지 않는 파일 모두 삭제 ===
# echo "🧹 '${KEYWORD}'를 포함하지 않는 파일 삭제 중..."
# find . -type f ! -name "*${KEYWORD}*" -delete
echo "✅ DONE! check pcd folder!"