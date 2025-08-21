#!/bin/bash

FOLDER_ID="1ph2ThjNAwUESrbsZkZr9rfx4y4qvxBMy"
OUTPUT_DIR="third_party"
FILE_NAME="mf_perception_third.tar"

echo "[Downloads] Google Drive ...."
gdown --folder "https://drive.google.com/drive/folders/${FOLDER_ID}" -O "${OUTPUT_DIR}"

cd "${OUTPUT_DIR}" || { echo "❌ fails... change directory"; exit 1; }

chmod 777 *
tar -xvf "${FILE_NAME}"

rm -rf "${FILE_NAME}"

echo "✅ DONE! check third_party folder!"