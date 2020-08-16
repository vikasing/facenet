export PYTHONPATH=/home/vikash/code/facenet/src
export LD_LIBRARY_PATH="/usr/local/cuda-10.1/lib64::$LD_LIBRARY_PATH"
nohup bash -c 'python src/train_softmax.py \
	--logs_base_dir ~/logs/facenet/ \
	--models_base_dir ~/models/facenet/ \
	--data_dir /data/vggface2/aligned/ \
	--image_size 160 \
	--model_def models.inception_resnet_v1 \
	--lfw_dir /data/lfw/aligned2/ \
	--optimizer ADAM \
	--learning_rate -1 \
	--max_nrof_epochs 500 \
	--batch_size 90 \
	--keep_probability 0.4 \
	--random_flip \
	--use_fixed_image_standardization \
	--learning_rate_schedule_file data/learning_rate_schedule_classifier_vggface2.txt \
	--weight_decay 5e-4 \
	--embedding_size 512 \
	--lfw_distance_metric 1 \
	--lfw_use_flipped_images \
	--lfw_subtract_mean \
	--validation_set_split_ratio 0.01 \
	--validate_every_n_epochs 5' &
