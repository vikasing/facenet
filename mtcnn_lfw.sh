#!/bin/sh
export PYTHONPATH=/home/vikash/code/facenet/src
#nohup python src/align/align_dataset_mtcnn.py /data/vggface2/train/ /data/vggface2/aligned/ --image_size 182 --margin 44&
nohup bash -c 'for N in {1..4}; do python src/align/align_dataset_mtcnn.py /data/lfw/raw/ /data/lfw/aligned2/ --image_size 160 --margin 32 --random_order --gpu_memory_fraction 0.25; done' &
