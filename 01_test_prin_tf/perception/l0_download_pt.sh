#!/bin/bash
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


FOLDER_ID="1Ly_eElD_ywGnZnPpuSxlpiSpECmaHHMf"
OUTPUT_DIR="ai_models"

echo "[Downloads] Google Drive pt...."
mkdir -p "${OUTPUT_DIR}"
gdown --folder "https://drive.google.com/drive/folders/${FOLDER_ID}" -O "${OUTPUT_DIR}"

cd "${OUTPUT_DIR}" || { echo "❌ 폴더 이동 실패"; exit 1; }

echo "✅ DONE! check pt folder!"


