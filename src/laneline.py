import os
import time
import cv2
import torch
import numpy as np
from skimage.morphology import skeletonize

from utils.yolopv2 import (
    select_device, driving_area_mask, lane_line_mask, LoadImages
)

def parse_args():
    args = {}
    args['weights'] = "./checkpoints/yolopv2.pt"
    args['img_size'] = 640
    args['conf_thres'] = 0.3
    args['iou_thres'] = 0.45
    args['device'] = "0"
    args['num_samples'] = 300
    return args

def get_mask(img_path, model, args, device, half):
    dataset = LoadImages(img_path, img_size=args['img_size'], stride=32)
    if device.type != 'cpu':
        model(torch.zeros(1, 3, args['img_size'], args['img_size']).to(device).type_as(next(model.parameters())))

    da_seg_masks = []
    ll_seg_masks = []
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        _, seg, ll = model(img)
        
        da_seg_mask = driving_area_mask(seg)
        ll_seg_mask = lane_line_mask(ll)

        da_seg_masks.append(da_seg_mask)
        ll_seg_masks.append(ll_seg_mask)
    
    return da_seg_masks, ll_seg_masks

def sample_skeleton_points(mask, num_points=300):
    skeleton = skeletonize(mask > 0)
    coords = np.column_stack(np.where(skeleton))  # (y, x)
    if len(coords) == 0:
        return np.empty((0, 2), dtype=np.float32)
    if len(coords) > num_points:
        idxs = np.linspace(0, len(coords) - 1, num_points, dtype=int)
        coords = coords[idxs]
    return coords[:, ::-1].astype(np.float32)  # return (x, y)

def compute_depth_sgbm(left_img, right_img, f, B, c, output_path):
    min_disp = 0
    num_disp = 128
    block_size = 5

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

    disp_norm = np.nan_to_num(disparity, nan=0.0)
    disp_norm[disp_norm < min_disp] = 0
    disp_vis = ((disp_norm - min_disp) / num_disp * 255).clip(0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(output_path, "disparity.png"), disp_vis)

    log_depth = np.log(depth_map + 1)
    log_depth = (log_depth / np.nanmax(log_depth) * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(output_path, "log_depth_map.png"), log_depth)

    return depth_map, disparity

def get_ll_points3d(left_path, right_path, P_left, P_right, f, B, c, output_path):
    args = parse_args()
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(os.path.join(output_path, "laneline"), exist_ok=True)

    # Model setup
    device = select_device(args['device'])
    half = device.type != 'cpu'
    model = torch.jit.load(args['weights']).to(device)
    if half:
        model.half()
    model.eval()

    with torch.no_grad():
        _, left_ll_masks = get_mask(left_path, model, args, device, half)
        _, right_ll_masks = get_mask(right_path, model, args, device, half)

    left_ll_mask, right_ll_mask = left_ll_masks[0], right_ll_masks[0]
    left_img = cv2.imread(left_path, cv2.IMREAD_COLOR)
    right_img = cv2.imread(right_path, cv2.IMREAD_COLOR)

    # Resize masks
    left_ll_mask = cv2.resize(left_ll_mask.astype(np.uint8), (left_img.shape[1], left_img.shape[0]), interpolation=cv2.INTER_NEAREST)
    right_ll_mask = cv2.resize(right_ll_mask.astype(np.uint8), (right_img.shape[1], right_img.shape[0]), interpolation=cv2.INTER_NEAREST)

    cv2.imwrite(os.path.join(output_path, "laneline", "left_lane_mask.png"), (left_ll_mask * 255).astype(np.uint8))
    cv2.imwrite(os.path.join(output_path, "laneline", "right_lane_mask.png"), (right_ll_mask * 255).astype(np.uint8))

    # Skeleton sampling
    left_pts = sample_skeleton_points(left_ll_mask, args['num_samples'])

    left_skel_img = left_img.copy()
    for x, y in left_pts.astype(int):
        if 0 <= x < left_skel_img.shape[1] and 0 <= y < left_skel_img.shape[0]:
            cv2.circle(left_skel_img, (x, y), 2, (0, 255, 255), -1)
    cv2.imwrite(os.path.join(output_path, "laneline", "left_skeleton.png"), left_skel_img)

    # Compute disparity and depth map
    depth_map, disparity_map = compute_depth_sgbm(left_img, right_img, f, B, c, os.path.join(output_path, "laneline"))

    # Project lane line points to 3D
    valid_pts = []
    pts_3d = []
    for pt in left_pts:
        x, y = int(pt[0]), int(pt[1])
        if 0 <= x < depth_map.shape[1] and 0 <= y < depth_map.shape[0]:
            z = depth_map[y, x]
            if z > 0:
                X = (x - c[0]) * z / f
                Y = (y - c[1]) * z / f
                pts_3d.append([X, Y, z])
                valid_pts.append([x, y])

    pts_3d = np.array(pts_3d)
    valid_pts = np.array(valid_pts)

    # Annotate
    for (x, y), z in zip(valid_pts, pts_3d[:, 2]):
        cv2.putText(left_img, f"{(z/100):.2f}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
        cv2.circle(left_img, (x, y), 3, (0, 255, 0), -1)
    cv2.imwrite(os.path.join(output_path, "laneline", "left_depth.png"), left_img)

    return pts_3d