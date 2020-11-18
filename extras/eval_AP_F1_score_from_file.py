import os
import pickle
import numpy as np
import sys
sys.path.append(".")
import shapely
from shapely.geometry import Polygon, MultiPoint
from evaluation.mean_average_precision import MetricBuilder
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET
from data import CARPLATE_FOUR_CORNERS_CLASSES as labelmap

devkit_path = '/data/CCPD/VOC/ccpd_db/'
set_type = 'test'
iou_thres = '0.5,0.75'
annopath = os.path.join(devkit_path, 'Annotations', '%s.xml')
imgsetpath = os.path.join(devkit_path, 'ImageSets',
                          'Main', '{:s}.txt')
bbox_cal_types = ['bbox', 'bbox_from_fc']
four_corners_cal_types = ['fc', 'fc_from_bbox']


def parse_rec(filename, object_size):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        # difficult always 0
        obj_struct['difficult'] = 0
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text) - 1,
                              int(bbox.find('ymin').text) - 1,
                              int(bbox.find('xmax').text) - 1,
                              int(bbox.find('ymax').text) - 1]
        obj_struct['four_corners'] = [int(bbox.find('x_top_left').text) - 1,
                              int(bbox.find('y_top_left').text) - 1,
                              int(bbox.find('x_top_right').text) - 1,
                              int(bbox.find('y_top_right').text) - 1,
                              int(bbox.find('x_bottom_right').text) - 1,
                              int(bbox.find('y_bottom_right').text) - 1,
                              int(bbox.find('x_bottom_left').text) - 1,
                              int(bbox.find('y_bottom_left').text) - 1]

        x_top_left = int(bbox.find('x_top_left').text) - 1
        y_top_left = int(bbox.find('y_top_left').text) - 1
        x_bottom_left = int(bbox.find('x_bottom_left').text) - 1
        y_bottom_left = int(bbox.find('y_bottom_left').text) - 1
        x_bottom_right = int(bbox.find('x_bottom_right').text) - 1
        y_bottom_right = int(bbox.find('y_bottom_right').text) - 1
        QP = np.array([x_top_left - x_bottom_left, y_top_left - y_bottom_left])
        v = np.array([x_bottom_left - x_bottom_right, y_bottom_left - y_bottom_right])
        h = np.linalg.norm(np.cross(QP, v))/np.linalg.norm(v)
        if object_size != 'all':
            if (h <= 16 and object_size != 'small') or (h > 16 and h <= 32 and object_size != 'medium') or (h > 32 and object_size != 'large'):
                continue

        objects.append(obj_struct)

    return objects


def polygon_iou(list1, list2):
    """
    Intersection over union between two shapely polygons.
    """
    polygon_points1 = np.array(list1).reshape(4, 2)
    poly1 = Polygon(polygon_points1).convex_hull
    polygon_points2 = np.array(list2).reshape(4, 2)
    poly2 = Polygon(polygon_points2).convex_hull
    union_poly = np.concatenate((polygon_points1, polygon_points2))
    if not poly1.intersects(poly2): # this test is fast and can accelerate calculation
        iou = 0
    else:
        try:
            inter_area = poly1.intersection(poly2).area
            #union_area = poly1.area + poly2.area - inter_area
            union_area = MultiPoint(union_poly).convex_hull.area
            if union_area == 0:
                return 0
            iou = float(inter_area) / union_area
        except shapely.geos.TopologicalError:
            print('shapely.geos.TopologicalError occured, iou set to 0')
            iou = 0
    return iou


def change_four_corners_to_bbox(four_corners_list):
    x_top_left, y_top_left, x_top_right, y_top_right,\
        x_bottom_right, y_bottom_right, x_bottom_left, y_bottom_left = four_corners_list

    xmin = min(x_top_left, x_bottom_left)
    ymin = min(y_top_left, y_top_right)
    xmax = max(x_bottom_right, x_top_right)
    ymax = max(y_bottom_right, y_bottom_left)

    return [xmin, ymin, xmax, ymax]


def change_bbox_to_four_corners(bbox_list):
    xmin, ymin, xmax, ymax = bbox_list

    x_top_left = xmin
    y_top_left = ymin
    x_top_right = xmax
    y_top_right = ymin
    x_bottom_right = xmax
    y_bottom_right = ymax
    x_bottom_left = xmin
    y_bottom_left = ymax

    return [x_top_left, y_top_left, x_top_right, y_top_right,
                x_bottom_right, y_bottom_right, x_bottom_left, y_bottom_left]


def calculate_AP(ovthresh, recs, detpath, classname, imagenames, cal_type='bbox'):
    # read dets
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()
    if any(lines) == 1:
        # exchange dets as key-value dic
        # 1. pred bbox with gt bbox; 2. bbox from pred four corners with gt bbox
        preds_dic = {}
        splitlines = [x.strip().split(' ') for x in lines]
        for x in splitlines:
            bbox_used = None
            if cal_type == 'bbox':
                bbox_used = x[2:6]
            elif cal_type == 'bbox_from_fc':
                bbox_used = change_four_corners_to_bbox(x[6:14])
            else:
                assert False, "wrong AP calculation type"
            if x[0] in preds_dic.keys():
                preds_dic[x[0]] = np.vstack((preds_dic[x[0]], np.append(np.array([float(z) for z in bbox_used]), [int(0), float(x[1])])))
            else:
                preds_dic[x[0]] = np.append(np.array([float(z) for z in bbox_used]), [int(0), float(x[1])])
                preds_dic[x[0]] = np.expand_dims(preds_dic[x[0]], axis=0)

        # create metric
        metric_fn = MetricBuilder.build_evaluation_metric("map_2d", async_mode=True, num_classes=1)
        # extract gt objects for this class
        for imagename in imagenames:
            R = [obj for obj in recs[imagename] if obj['name'] == classname]
            GT_bbox = np.array([x['bbox'] for x in R])
            GT_append = np.zeros((GT_bbox.shape[0], 3), dtype=np.int)
            GTs = np.hstack((GT_bbox, GT_append))
            if imagename not in preds_dic.keys():
                preds = np.array([[]])
            else:
                preds = preds_dic[imagename]
            metric_fn.add(preds, GTs)

        metrics = metric_fn.value(iou_thresholds=ovthresh)
        rec = metrics[ovthresh][0]['recall']
        prec = metrics[ovthresh][0]['precision']
        ap = metrics[ovthresh][0]['ap']

    return rec, prec, ap


def calculate_F1_score(ovthresh, recs, detpath, classname, cal_type='fc'):
    class_recs = {}
    npos = 0
    for imagename in recs.keys():
        R = [obj for obj in recs[imagename] if obj['name'] == classname]
        four_corners = np.array([x['four_corners'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {'four_corners': four_corners,
                                'difficult': difficult,
                                'det': det}

    # read dets
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()
    if any(lines) == 1:
        splitlines = [x.strip().split(' ') for x in lines]
        image_ids = [x[0] for x in splitlines]
        confidence = np.array([float(x[1]) for x in splitlines])
        conf_idx = confidence > 0.5
        image_ids = np.array(image_ids)[conf_idx]
        image_ids = image_ids.tolist()
        confidence = confidence[conf_idx]
        BB = np.array([[float(z) for z in x[2:]] for x in splitlines])
        BB = BB[conf_idx, :]

        # sort by confidence
        sorted_ind = np.argsort(-confidence)
        BB = BB[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]

        # go down dets and mark TPs and FPs
        nd = len(image_ids)
        tp = np.zeros(nd)
        fp = np.zeros(nd)
        for d in range(nd):
            R = class_recs[image_ids[d]]
            bb = BB[d, :].astype(float)
            ovmax = -np.inf
            BBGT = R['four_corners'].astype(float)
            if cal_type == 'fc':
                bb = bb[4:]
            elif cal_type == 'fc_from_bbox':
                bb = np.array(change_bbox_to_four_corners(bb[:4]))
            else:
                assert False, "wrong f1 calculation type"
            if BBGT.size > 0:
                # compute overlaps
                overlaps = np.zeros(BBGT.shape[0])
                for idx in range(BBGT.shape[0]):
                    overlap = polygon_iou(bb.tolist(), BBGT[idx, :].tolist())
                    overlaps[idx] = overlap
                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)

            if ovmax > ovthresh:
                if not R['difficult'][jmax]:
                    if not R['det'][jmax]:
                        tp[d] = 1.
                        R['det'][jmax] = 1
                    else:
                        fp[d] = 1.
            else:
                fp[d] = 1.

        tp_num = np.sum(tp)
        fp_num = np.sum(fp)
        rec = tp_num / float(npos)
        prec = tp_num / float(tp_num + fp_num)
        F1 = 2.*prec*rec/(prec+rec)

    return rec, prec, F1


def eval_AP(detpath,
            annopath,
            imagesetfile,
            classname,
            cachedir,
            cal_type,
            ovthresh=0.5,
            use_12_metric=True,
            object_size='all'):
    # first load gt
    if not os.path.isdir(cachedir):
        os.mkdir(cachedir)
    cachefile = os.path.join(cachedir, 'annots.pkl')
    # read list of images
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]
    if not os.path.isfile(cachefile):
        # load annots
        recs = {}
        for i, imagename in enumerate(imagenames):
            parse_results = parse_rec(annopath % (imagename), object_size)
            if parse_results == []:
                continue
            recs[imagename] = parse_results
            if i % 100 == 0:
                print('Reading annotation for {:d}/{:d}'.format(
                   i + 1, len(imagenames)))
        # save
        print('Saving cached annotations to {:s}'.format(cachefile))
        with open(cachefile, 'wb') as f:
            pickle.dump(recs, f)
    else:
        # load
        with open(cachefile, 'rb') as f:
            recs = pickle.load(f)

    rec, prec, ap = calculate_AP(ovthresh, recs, detpath, classname, imagenames, cal_type=cal_type)

    return rec, prec, ap


def eval_F1_score(detpath,
                  annopath,
                  imagesetfile,
                  classname,
                  cachedir,
                  cal_type,
                  ovthresh=0.5,
                  use_12_metric=True,
                  object_size='all'):
    # first load gt
    if not os.path.isdir(cachedir):
        os.mkdir(cachedir)
    cachefile = os.path.join(cachedir, 'annots.pkl')
    # read list of images
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]
    if not os.path.isfile(cachefile):
        # load annots
        recs = {}
        for i, imagename in enumerate(imagenames):
            parse_results = parse_rec(annopath % (imagename), object_size)
            if parse_results == []:
                continue
            recs[imagename] = parse_results
            if i % 100 == 0:
                print('Reading annotation for {:d}/{:d}'.format(
                   i + 1, len(imagenames)))
        # save
        print('Saving cached annotations to {:s}'.format(cachefile))
        with open(cachefile, 'wb') as f:
            pickle.dump(recs, f)
    else:
        # load
        with open(cachefile, 'rb') as f:
            recs = pickle.load(f)

    rec, prec, F1 = calculate_F1_score(ovthresh, recs, detpath, classname, cal_type=cal_type)

    return rec, prec, F1


def get_voc_results_file_template(image_set, cls):
    # VOCdevkit/VOC2007/results/det_test_aeroplane.txt
    filename = 'det_' + image_set + '_%s.txt' % (cls)
    filedir = os.path.join(devkit_path, 'results')
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    path = os.path.join(filedir, filename)
    return path


def do_python_eval(use_12=True, object_size='all'):
    cachedir = os.path.join(devkit_path, 'annotations_cache')
    # The PASCAL VOC metric changed in 2010
    use_12_metric = use_12
    print('VOC12 metric? ' + ('Yes' if use_12_metric else 'No'))
    for _, cls in enumerate(labelmap):
        filename = get_voc_results_file_template(set_type, cls)
        # AP for horizontal bbox
        if object_size == 'all':
            for cal_type in bbox_cal_types:
                for t in iou_thres.split(','):
                    rec, prec, ap = eval_AP(
                        filename, annopath, imgsetpath.format(set_type), cls, cachedir, cal_type,
                        ovthresh=float(t.strip()), use_12_metric=use_12_metric, object_size=object_size)
                    if cal_type == 'bbox':
                        print('AP for {} with IoU threshold {} = {:.4f}'.format(cls, t.strip(), ap))
                    elif cal_type == 'bbox_from_fc':
                        print('AP for {} with IoU threshold {} from four corners = {:.4f}'.format(cls, t.strip(), ap))
        # F1-score for quadarilateral bbox
        for cal_type in four_corners_cal_types:
            for t in iou_thres.split(','):
                rec, prec, F1 = eval_F1_score(
                    filename, annopath, imgsetpath.format(set_type), cls, cachedir, cal_type,
                    ovthresh=float(t.strip()), use_12_metric=use_12_metric, object_size=object_size)
                if cal_type == 'fc':
                    print('{} with IoU threshold {} => recall: {:.4f}, precision: {:.4f}, F1-score: {:.4f}'.format(
                        cls, t.strip(), rec, prec, F1))
                elif cal_type == 'fc_from_bbox':
                    print('{} with IoU threshold {} from bbox => recall: {:.4f}, precision: {:.4f}, F1-score: {:.4f}'.format(
                        cls, t.strip(), rec, prec, F1))
    print('--------------------------------------------------------------')
    print('Results computed with the **unofficial** Python eval code.')
    print('refer to https://github.com/bes-dev/mean_average_precision')
    print('--------------------------------------------------------------')


if __name__ == '__main__':
    do_python_eval(use_12=True, object_size='all')