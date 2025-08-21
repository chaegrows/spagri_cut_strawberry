xhost +

#echo "rm is enabled"
docker run  \
  --rm \
  -it \
  --name dset_preprocessing \
  -v ./:/root/dset_preprocessing \
  -v ./mot_out:/tmp/mot_out \
  --workdir /root/dset_preprocessing \
  dset_preprocessing:250120 \
  python3 extract_images.py data/sj_straw_room1_formap 
