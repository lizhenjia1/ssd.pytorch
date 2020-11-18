import os

root_dir = '/data/CCPD/test/'
datasets = os.listdir(root_dir)
models = {'CCPD_carplate_bbox_weights/ssd300_45000.pth': ['carplate', 300],
        'CCPD_carplate_bbox_weights/ssd512_50000.pth': ['carplate', 512],
        'CCPD_carplate_only_four_corners_weights/CARPLATE_ONLY_FOUR_CORNERS300.pth': ['carplate_only_four_corners', 300],
        'CCPD_carplate_only_four_corners_weights/ssd512_55000.pth': ['carplate_only_four_corners', 512],
        'CCPD_carplate_bbox_four_corners_weights/ssd300_35000.pth': ['carplate_four_corners', 300],
        'CCPD_carplate_bbox_four_corners_weights/ssd512_35000.pth': ['carplate_four_corners', 512],
        'CCPD_carplate_bbox_four_corners_with_border_loss_weights/CARPLATE_FOUR_CORNERS_WITH_BORDER300.pth': ['carplate_four_corners', 300],
        'CCPD_carplate_bbox_four_corners_with_border_loss_weights/ssd512_20000.pth': ['carplate_four_corners', 512]}

for k, v in models.items():
    for dataset in datasets:
        print('CUDA_VISIBLE_DEVICES=1 python evaluation/eval_carplate_four_corners.py --trained_model {} --voc_root {} --input_size {} --obj_type {}'.format(
            'weights/'+k, root_dir+dataset+'/', v[1], v[0]))
        os.system('CUDA_VISIBLE_DEVICES=1 python evaluation/eval_carplate_four_corners.py --trained_model {} --voc_root {} --input_size {} --obj_type {}'.format(
            'weights/'+k, root_dir+dataset+'/', v[1], v[0]))
