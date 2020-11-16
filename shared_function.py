import shapely
from shapely.geometry import Polygon, MultiPoint
import numpy as np


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
