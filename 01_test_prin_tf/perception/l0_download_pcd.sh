#!/bin/bash
# pip install gdown

# Check if gdown is installed, if not install it
if ! command -v gdown &> /dev/null; then
    echo "gdown not found. Installing..."
    pip install gdown
    # Add user local bin to PATH if not already there
    export PATH="$HOME/.local/bin:$PATH"
fi

# Ensure gdown is available in PATH
if ! command -v gdown &> /dev/null; then
    # Try using python module directly
    GDOWN_CMD="python -m gdown"
else
    GDOWN_CMD="gdown"
fi

FOLDER_ID="10ecduyOh_xu8TdY3rT-C2bArGpK355rG"
KEYWORD=${1:-"isu_sj_straw"}  # 인자로 받되, 없으면 기본값 사용
OUTPUT_DIR="pcd"

echo "[Downloads] Google Drive pcd...."
$GDOWN_CMD --folder "https://drive.google.com/drive/folders/${FOLDER_ID}" -O "${OUTPUT_DIR}"

cd "${OUTPUT_DIR}" || { echo "❌ 폴더 이동 실패"; exit 1; }
# === 키워드를 포함하지 않는 파일 모두 삭제 ===
# echo "🧹 '${KEYWORD}'를 포함하지 않는 파일 삭제 중..."
# find . -type f ! -name "*${KEYWORD}*" -delete
echo "✅ DONE! check pcd folder!"