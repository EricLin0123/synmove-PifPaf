import cv2
import numpy as np

def compute_depth_sgbm(left_img, right_img, f, B, min_disp=0, num_disp=128, block_size=5):
    """
    Compute disparity map using StereoSGBM.
    """
    stereo = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=block_size,
        P1=8 * 3 * block_size ** 2,
        P2=32 * 3 * block_size ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=2,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

    disparity = stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0
    disparity[disparity <= 0.1] = np.nan

    depth_map = (f * B) / disparity
    depth_map[np.isnan(depth_map)] = 0

    return depth_map, disparity

def ll_points3d(pts, depth_map, f, c):
    """
    Convert 2D points to 3D points using disparity.
    """
    valid_pts = []
    pts_3d = []
    for pt in pts:
        x, y = int(pt[0]), int(pt[1])
        if 0 <= x < depth_map.shape[1] and 0 <= y < depth_map.shape[0]:
            z = depth_map[y, x]
            if z > 0:
                X = (x - c[0]) * z / f
                Y = (y - c[1]) * z / f
                pts_3d.append([X, Y, z])
                valid_pts.append([x, y])

    return pts_3d, valid_pts

def ll_point3d_to_bev2d(point3d_list):
    """
    Convert 3D points to BEV 2D points.
    """
    ll_bev2d = []
    for pt in point3d_list:
        x, _, z = pt
        ll_bev2d.append((x, z))

    return ll_bev2d