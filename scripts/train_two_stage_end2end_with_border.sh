CUDA_VISIBLE_DEVICES=3 python train_two_stage_end2end_with_border.py --dataset TWO_STAGE_END2END --dataset_root /data/VALID/720p/car_carplate_two_stage_end2end/VOC/ --save_folder two_stage_end2end_with_border_weights/ --input_size 512 --lr 1e-4 --visdom True --batch_size 10