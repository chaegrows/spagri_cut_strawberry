xhost +

#echo "rm is enabled"
docker run  \
  --rm \
  -it \
  --name dset_preprocessing \
  -v ./:/root/dset_preprocessing \
  --workdir /root/dset_preprocessing \
  dset_preprocessing:250120 \
  /bin/bash  
