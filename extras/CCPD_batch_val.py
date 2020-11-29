import os

root_dir = '/data/CCPD/VOC/'
models = {
    'CCPD_carplate_bbox_weights': ['carplate', 512],
    'CCPD_carplate_bbox_four_corners_weights': ['carplate_four_corners', 512],
    'CCPD_carplate_bbox_four_corners_with_CIoU_loss_weights': ['carplate_four_corners', 512],
    'CCPD_carplate_only_four_corners_weights': ['carplate_only_four_corners', 512],
    'CCPD_carplate_only_four_corners_with_IoU_loss_weights': ['carplate_only_four_corners', 512],
    'CCPD_carplate_only_four_corners_with_GIoU_loss_weights': ['carplate_only_four_corners', 512],
    'CCPD_carplate_only_four_corners_with_DIoU_loss_weights': ['carplate_only_four_corners', 512],
    'CCPD_carplate_only_four_corners_with_border_loss_weights': ['carplate_only_four_corners', 512],
    'CCPD_carplate_only_four_corners_with_CIoU_loss_weights_16': ['carplate_only_four_corners', 512],
        }

for k, v in models.items():
    for pth in os.listdir(os.path.join('weights', k)):
        if '512' in pth:
            print('CUDA_VISIBLE_DEVICES=0 python evaluation/eval_carplate_four_corners.py --trained_model {} --voc_root {} --input_size {} --obj_type {} --iou_thres {}'.format(
                os.path.join('weights', k, pth), root_dir, v[1], v[0], '0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95'))
            os.system('CUDA_VISIBLE_DEVICES=0 python evaluation/eval_carplate_four_corners.py --trained_model {} --voc_root {} --input_size {} --obj_type {} --iou_thres {}'.format(
                os.path.join('weights', k, pth), root_dir, v[1], v[0], '0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95'))
