import os

root_dir = '/data/CCPD/test/'
datasets = os.listdir(root_dir)
models = {
    # 'CCPD_carplate_bbox_weights/ssd512_40000.pth': ['carplate', 512],
    # 'CCPD_carplate_only_four_corners_weights/ssd512_40000.pth': ['carplate_only_four_corners', 512],
    # 'CCPD_carplate_bbox_four_corners_weights/ssd512_35000.pth': ['carplate_four_corners', 512],
    # 'CCPD_carplate_bbox_four_corners_with_CIoU_loss_weights_16/ssd512_50000.pth': ['carplate_four_corners', 512],
    # 'CCPD_carplate_only_four_corners_with_IoU_loss_weights_16/ssd512_50000.pth': ['carplate_only_four_corners', 512],
    # 'CCPD_carplate_only_four_corners_with_GIoU_loss_weights_16/CARPLATE_ONLY_FOUR_CORNERS_WITH_BORDER512.pth': ['carplate_only_four_corners', 512],
    # 'CCPD_carplate_only_four_corners_with_DIoU_loss_weights_16/ssd512_55000.pth': ['carplate_only_four_corners', 512],
    # 'CCPD_carplate_only_four_corners_with_border_loss_weights_16/ssd512_25000.pth': ['carplate_only_four_corners', 512],
    # 'CCPD_carplate_only_four_corners_with_CIoU_loss_weights_1/ssd512_55000.pth': ['carplate_only_four_corners', 512],
    # 'CCPD_carplate_only_four_corners_with_CIoU_loss_weights_3/ssd512_35000.pth': ['carplate_only_four_corners', 512],
    # 'CCPD_carplate_only_four_corners_with_CIoU_loss_weights_5/ssd512_55000.pth': ['carplate_only_four_corners', 512],
    # 'CCPD_carplate_only_four_corners_with_CIoU_loss_weights_10/ssd512_25000.pth': ['carplate_only_four_corners', 512],
    # 'CCPD_carplate_only_four_corners_with_CIoU_loss_weights_16/ssd512_50000.pth': ['carplate_only_four_corners', 512],
    # 'CCPD_carplate_only_four_corners_with_CIoU_loss_weights_30/ssd512_45000.pth': ['carplate_only_four_corners', 512],
    # 'CCPD_carplate_only_four_corners_with_CIoU_loss_weights_50/ssd512_40000.pth': ['carplate_only_four_corners', 512],
    # 'CCPD_carplate_only_four_corners_with_CIoU_loss_weights_100/ssd512_50000.pth': ['carplate_only_four_corners', 512],
    # 'CCPD_carplate_only_four_corners++_weights/ssd512_50000.pth': ['carplate_only_four_corners', 512],
    # 'CCPD_carplate_bbox_four_corners++_weights/CARPLATE_FOUR_CORNERS512.pth': ['carplate_four_corners', 512],
    # 'CCPD_carplate_bbox_four_corners++_with_CIoU_loss_weights_16/ssd512_25000.pth': ['carplate_four_corners', 512],
    # 'CCPD_carplate_only_four_corners++_with_CIoU_loss_weights_16/ssd512_50000.pth': ['carplate_only_four_corners', 512],
    # 'CCPD_carplate_bbox_weights/ssd300_50000.pth': ['carplate', 300],
    # 'CCPD_carplate_only_four_corners_weights/ssd300_50000.pth': ['carplate_only_four_corners', 300],
    # 'CCPD_carplate_bbox_four_corners_weights/ssd300_50000.pth': ['carplate_four_corners', 300],
    # 'CCPD_carplate_bbox_four_corners_with_CIoU_loss_weights_16/ssd300_45000.pth': ['carplate_four_corners', 300],
    # 'CCPD_carplate_only_four_corners_with_CIoU_loss_weights_16/ssd300_35000.pth': ['carplate_only_four_corners', 300],
    # 'CCPD_carplate_only_four_corners++_weights/CARPLATE_ONLY_FOUR_CORNERS300.pth': ['carplate_only_four_corners', 300],
    # 'CCPD_carplate_bbox_four_corners++_weights/ssd300_50000.pth': ['carplate_four_corners', 300],
    # 'CCPD_carplate_bbox_four_corners++_with_CIoU_loss_weights_16/ssd300_55000.pth': ['carplate_four_corners', 300],
    # 'CCPD_carplate_only_four_corners++_with_CIoU_loss_weights_16/ssd300_25000.pth': ['carplate_only_four_corners', 300],
        }

for k, v in models.items():
    for dataset in datasets:
        print('CUDA_VISIBLE_DEVICES=3 python evaluation/eval_carplate_four_corners.py --trained_model {} --voc_root {} --input_size {} --obj_type {} --iou_thres {}'.format(
            'weights/'+k, root_dir+dataset+'/', v[1], v[0], '0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95'))
        os.system('CUDA_VISIBLE_DEVICES=3 python evaluation/eval_carplate_four_corners.py --trained_model {} --voc_root {} --input_size {} --obj_type {} --iou_thres {}'.format(
            'weights/'+k, root_dir+dataset+'/', v[1], v[0], '0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95'))
