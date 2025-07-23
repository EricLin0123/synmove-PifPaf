import numpy as np

def car_keypoints3d(ann_left, ann_right, focal_length:tuple, baseline:float, centriod_left:tuple):
    """
    Estimate car's 3D position from matched objects.
    """
    keypoints_l = ann_left.data[:, :3]  # shape (K, 3): x, y, conf
    keypoints_r = ann_right.data[:, :3]

    keypoints_l = ann_left.data[:, :3]  # x, y, conf
    keypoints_r = ann_right.data[:, :3]

    points_3d = []
    for (xl, yl, cl), (xr, _, cr) in zip(keypoints_l, keypoints_r):
        if cl > 0.1 and cr > 0.1:
            disparity = xl - xr
            if disparity != 0:
                Z = focal_length[0] * baseline / disparity
                X = (xl - centriod_left[0]) * Z / focal_length[0]
                Y = (yl - centriod_left[1]) * Z / focal_length[1]
                points_3d.append((X, Y, Z))
            else:
                points_3d.append(None)
        else:
            points_3d.append(None)
    return points_3d

def car_point3d_to_bev2d(point3d_list):
    """
    Convert 3D points to 2D BEV points.
    """
    bev2d_list = []
    for pts in point3d_list:
        bev_pts = []
        for pt in pts:
            if pt is not None:
                x, _, z = pt
                bev_pts.append((x, z))
            else:
                bev_pts.append(None)
        bev2d_list.append(bev_pts)
    return bev2d_list