CUDA_VISIBLE_DEVICES=3 python train.py --dataset CAR --dataset_root /data/VALID/720p/car_only/VOC --save_folder car_weights/ --input_size 300 --lr 1e-4 --visdom True --batch_size 10 --set_type car