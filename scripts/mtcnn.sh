#!/bin/sh
export PYTHONPATH=/home/vikash/code/facenet/src
export LD_LIBRARY_PATH="/usr/local/cuda-10.1/lib64::$LD_LIBRARY_PATH"
#nohup python src/align/align_dataset_mtcnn.py /data/vggface2/train/ /data/vggface2/aligned/ --image_size 182 --margin 44&
nohup bash -c 'for N in {1..100}; do python src/align/align_dataset_mtcnn.py /data/vggface2/train/ /data/vggface2/aligned/ --image_size 182 --margin 44 --random_order --gpu_memory_fraction 0.1; done' &
