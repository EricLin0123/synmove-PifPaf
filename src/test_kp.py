import os
import argparse
import time
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from skimage.morphology import skeletonize
from skimage.util import img_as_ubyte

from utils.yolopv2 import (
    time_synchronized, select_device, increment_path,
    scale_coords, xyxy2xywh, non_max_suppression, split_for_trace_model,
    driving_area_mask, lane_line_mask, plot_one_box, show_seg_result,
    AverageMeter, LoadImages
)


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='checkpoints/yolopv2.pt')
    parser.add_argument('--source', type=str, default='sample_data/')
    parser.add_argument('--img-size', type=int, default=640)
    parser.add_argument('--conf-thres', type=float, default=0.3)
    parser.add_argument('--iou-thres', type=float, default=0.45)
    parser.add_argument('--device', default='0')
    parser.add_argument('--output', type=str, default='outputs')
    parser.add_argument('--match-threshold', type=float, default=50.0)
    parser.add_argument('--num-samples', type=int, default=300)
    parser.add_argument('--epipolar-tol', type=float, default=2.0)
    return parser


def load_projection_matrices(calib_file):
    P_rect_02, P_rect_03 = None, None
    with open(calib_file, 'r') as f:
        for line in f:
            if line.startswith('P_rect_02'):
                P_rect_02 = np.array([float(x) for x in line.split()[1:]]).reshape(3, 4)
            elif line.startswith('P_rect_03'):
                P_rect_03 = np.array([float(x) for x in line.split()[1:]]).reshape(3, 4)
    return P_rect_02, P_rect_03


def get_mask(img_path, model, opt, device, half):
    dataset = LoadImages(img_path, img_size=opt.img_size, stride=32)
    if device.type != 'cpu':
        model(torch.zeros(1, 3, opt.img_size, opt.img_size).to(device).type_as(next(model.parameters())))

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


if __name__ == '__main__':
    opt = make_parser().parse_args()
    os.makedirs(opt.output, exist_ok=True)

    # Paths
    left_path = os.path.join(opt.source, 'left.png')
    right_path = os.path.join(opt.source, 'right.png')
    calib_file = os.path.join(opt.source, 'calib.txt')

    # Model setup
    device = select_device(opt.device)
    half = device.type != 'cpu'
    model = torch.jit.load(opt.weights).to(device)
    if half:
        model.half()
    model.eval()

    # Lane masks
    with torch.no_grad():
        _, left_ll_masks = get_mask(left_path, model, opt, device, half)
        _, right_ll_masks = get_mask(right_path, model, opt, device, half)

    left_ll_mask, right_ll_mask = left_ll_masks[0], right_ll_masks[0]
    left_img = cv2.imread(left_path, cv2.IMREAD_COLOR)
    right_img = cv2.imread(right_path, cv2.IMREAD_COLOR)

    # Resize lane line masks to original image size
    left_ll_mask = cv2.resize(left_ll_mask.astype(np.uint8), (left_img.shape[1], left_img.shape[0]), interpolation=cv2.INTER_NEAREST)
    right_ll_mask = cv2.resize(right_ll_mask.astype(np.uint8), (right_img.shape[1], right_img.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Save masks
    cv2.imwrite(os.path.join(opt.output, "left_lane_mask.png"), (left_ll_mask * 255).astype(np.uint8))
    cv2.imwrite(os.path.join(opt.output, "right_lane_mask.png"), (right_ll_mask * 255).astype(np.uint8))

    # Skeleton sampling
    left_pts = sample_skeleton_points(left_ll_mask, opt.num_samples)
    right_pts = sample_skeleton_points(right_ll_mask, opt.num_samples)

    # Show and save skeleton overlay
    left_skel_img = left_img.copy()
    right_skel_img = right_img.copy()

    for x, y in left_pts.astype(int):
        if 0 <= x < left_skel_img.shape[1] and 0 <= y < left_skel_img.shape[0]:
            cv2.circle(left_skel_img, (x, y), 2, (0, 255, 255), -1)

    for x, y in right_pts.astype(int):
        if 0 <= x < right_skel_img.shape[1] and 0 <= y < right_skel_img.shape[0]:
            cv2.circle(right_skel_img, (x, y), 2, (0, 255, 255), -1)

    cv2.imwrite(os.path.join(opt.output, "left_skeleton.png"), left_skel_img)
    cv2.imwrite(os.path.join(opt.output, "right_skeleton.png"), right_skel_img)

    if len(left_pts) == 0 or len(right_pts) == 0:
        raise RuntimeError("No skeleton points found.")

    # Pixel distance cost matrix
    cost_matrix = np.linalg.norm(left_pts[:, None, :] - right_pts[None, :, :], axis=2)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    good_matches = [(i, j) for i, j in zip(row_ind, col_ind) if cost_matrix[i, j] < opt.match_threshold]

    epipolar_valid_matches = [
        (i, j) for (i, j) in good_matches
        if abs(left_pts[i][1] - right_pts[j][1]) < opt.epipolar_tol
    ]

    if not epipolar_valid_matches:
        raise RuntimeError("No matches passed epipolar constraint.")

    # Triangulation
    matched_left = np.array([left_pts[i] for i, _ in epipolar_valid_matches]).T
    matched_right = np.array([right_pts[j] for _, j in epipolar_valid_matches]).T

    P_rect_02, P_rect_03 = load_projection_matrices(calib_file)
    pts_4d_hom = cv2.triangulatePoints(P_rect_02, P_rect_03, matched_left, matched_right)
    pts_3d = pts_4d_hom[:3] / pts_4d_hom[3]
    depths = pts_3d[2]

    # Only use points with positive depth
    valid_mask = depths > 0
    valid_pts_3d = pts_3d[:, valid_mask]
    valid_matched_left = matched_left[:, valid_mask]
    valid_depths = depths[valid_mask]

    # Annotate depths
    for pt, depth in zip(valid_matched_left.T, valid_depths):
        x, y = int(pt[0]), int(pt[1])
        if 0 <= x < left_img.shape[1] and 0 <= y < left_img.shape[0]:
            cv2.putText(left_img, f"{depth:.2f}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
            cv2.circle(left_img, (x, y), 3, (0, 255, 0), -1)

    # Save annotated image
    cv2.imwrite(os.path.join(opt.output, "left_depth.png"), left_img)

    # 3D Plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(valid_pts_3d[0], valid_pts_3d[1], valid_pts_3d[2], c='b', marker='o')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Points from Triangulation')
    ax.view_init(vertical_axis='y', elev=0, azim=-90)
    plt.savefig(os.path.join(opt.output, "3d_points.png"))

    print("Processing complete.")