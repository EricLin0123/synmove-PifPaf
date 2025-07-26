""" Process multiple stereo image pairs for BEV construction. """
import os
import cv2
import glob
import torch
import logging
import argparse
import numpy as np
from tqdm import tqdm

# openpifpaf
from openpifpaf import decoder, logger, network, show, visualizer, __version__
from openpifpaf.predictor import Predictor

# utils
# general utils
from utils.common import load_projection_matrices
from utils.visualize import (
    bev, car_annotate, car_draw_matches, ll_draw_skeleton_points,
    ll_disparity_map, ll_log_depth_map, ll_annotate
)                                                                     # for all visualizations

# car model
# for matching vehicles between left and right images
from utils.car.matching import hungarian_centroid_match
from utils.car.keypoint import car_keypoints3d, car_point3d_to_bev2d  # for estimating 3D points and converting to BEV
from utils.car.registration import register_car_model                 # for matching keypoints with full car model
from utils.car.apollo_skeleton import apollo_skeleton24               # for car keypoints and skeleton

# lane line
from utils.lane.yolopv2 import select_device, get_mask                # for YOLOPv2 lane line detection
from utils.lane.sampling import sample_skeleton_points                # for skeletonized sampling of lane line points
from utils.lane.keypoint import (
    compute_depth_sgbm, ll_points3d, ll_point3d_to_bev2d
)                                                                     # for computing disparity and converting points to BEV
from utils.lane.cluster import (
    group_ll_dbscan, group_ll_kmeans, fit_lane_lines
)                                                                     # for clustering lane line points

LOG = logging.getLogger(__name__)


def cli():
    parser = argparse.ArgumentParser(
        prog='python3 -m main_stream',
        usage='%(prog)s [options] data_dir',
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--version', action='version',
                        version='OpenPifPaf {version}'.format(version=__version__))

    decoder.cli(parser)
    logger.cli(parser)
    network.Factory.cli(parser)
    Predictor.cli(parser)
    show.cli(parser)
    visualizer.cli(parser)

    # I/O parameters
    parser.add_argument('data_dir',
                        help='input data directory containing left/ and right/ subdirectories')
    parser.add_argument('--output', type=str, default="output_stream",
                        help='output directory')
    parser.add_argument('--calib', type=str, default=None,
                        help='calibration file path (if not provided, searches for calib.txt in data_dir)')
    parser.add_argument('--disable-cuda', action='store_true',
                        help='disable CUDA')

    # camera parameters
    parser.add_argument('--baseline', type=float, default=45,
                        help='baseline distance between left and right cameras (in cm)')

    # plotting parameters
    parser.add_argument('--bev_scale', type=float, default=1800,
                        help='scale for bird\'s eye view plotting')

    # car model registration parameters
    parser.add_argument('--car_scale', type=float, default=30,
                        help='car model scale')
    parser.add_argument('--ransac_iter', type=int, default=100,
                        help='number of RANSAC iterations for car model registration')
    parser.add_argument('--ransac_threshold', type=float, default=50,
                        help='RANSAC threshold for car model registration')

    # lane line detection & processing parameters
    parser.add_argument('--ckpt', type=str, default='checkpoints/yolopv2.pt',
                        help='path to the YOLOPv2 checkpoint for lane line detection')
    # YOLOPv2
    parser.add_argument('--img_size', type=int, default=640,
                        help='input image size for YOLOPv2')
    parser.add_argument('--conf_thres', type=float, default=0.3,
                        help='confidence threshold for YOLOPv2')
    parser.add_argument('--iou_thres', type=float, default=0.45,
                        help='IOU threshold for YOLOPv2')
    # skeletonized sampling
    parser.add_argument('--num_samples', type=int, default=300,
                        help='number of samples after skeletonized sampling')
    # SGBM depth estimation
    parser.add_argument('--min_disp', type=int, default=0,
                        help='minimum disparity for SGBM')
    parser.add_argument('--num_disp', type=int, default=128,
                        help='number of disparities for SGBM')
    parser.add_argument('--block_size', type=int, default=5,
                        help='block size for SGBM')
    # clustering
    parser.add_argument('--clustering_method', type=str, default='dbscan',
                        choices=['dbscan', 'kmeans'],
                        help='clustering method for lane line points')
    parser.add_argument('--eps', type=float, default=100,
                        help='epsilon for DBSCAN clustering')
    parser.add_argument('--min_samples', type=int, default=20,
                        help='minimum samples for DBSCAN clustering')
    parser.add_argument('--n_clusters', type=int, default=4,
                        help='number of clusters for KMeans clustering')
    parser.add_argument('--z_max', type=float, default=3000,
                        help='maximum z value for filtering points')
    parser.add_argument('--z_weight', type=float, default=0.2,
                        help='weight for z value in clustering')
    parser.add_argument('--use_z', action='store_true',
                        help='use z value in clustering (DBSCAN and KMeans)')
    # curve fitting
    parser.add_argument('--fit_type', type=str, default='linear',
                        choices=['linear', 'quadratic'],
                        help='type of curve fitting for lane lines')
    parser.add_argument('--line_conf', type=float, default=500,
                        help='confidence threshold for curve fitting')
    parser.add_argument('--num_line_points', type=int, default=100,
                        help='number of points for each lane line curve')

    args = parser.parse_args()

    logger.configure(args, LOG)  # logger first

    # add args.device
    args.device = torch.device('cpu')
    args.pin_memory = False
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        args.pin_memory = True
    LOG.info('neural network device: %s (CUDA available: %s, count: %d)',
             args.device, torch.cuda.is_available(), torch.cuda.device_count())

    decoder.configure(args)
    network.Factory.configure(args)
    Predictor.configure(args)
    show.configure(args)
    visualizer.configure(args)

    return args


def get_image_pairs(data_dir):
    """Get list of corresponding left and right image pairs."""
    left_dir = os.path.join(data_dir, 'left')
    right_dir = os.path.join(data_dir, 'right')

    if not os.path.exists(left_dir) or not os.path.exists(right_dir):
        raise RuntimeError(f"Expected 'left' and 'right' subdirectories in {data_dir}")

    # Get all PNG files from left directory
    left_files = sorted(glob.glob(os.path.join(left_dir, '*.png')))

    image_pairs = []
    for left_path in left_files:
        filename = os.path.basename(left_path)
        right_path = os.path.join(right_dir, filename)

        if os.path.exists(right_path):
            image_pairs.append((left_path, right_path, filename))
        else:
            LOG.warning(f"Missing right image for {filename}")

    return image_pairs


def process_single_pair(left_image_path, right_image_path, frame_name, args, predictor, yolo_model, device, half):
    """Process a single stereo image pair."""
    # Create output directory for this frame
    frame_output_dir = os.path.join(args.output, frame_name.replace('.png', ''))
    os.makedirs(frame_output_dir, exist_ok=True)
    os.makedirs(os.path.join(frame_output_dir, 'car'), exist_ok=True)
    os.makedirs(os.path.join(frame_output_dir, 'lane'), exist_ok=True)

    # Load calibration matrices
    calib_file_path = args.calib if args.calib else os.path.join(args.data_dir, 'calib.txt')
    if not os.path.exists(calib_file_path):
        LOG.error(f"Calibration file not found: {calib_file_path}")
        return False

    P_left, _ = load_projection_matrices(calib_file_path)
    fx = P_left[0, 0]  # focal length
    fy = P_left[1, 1]  # focal length
    c_left = [P_left[0, 2], P_left[1, 2]]  # principal point left

    try:
        # ========== Car Model Part ========== #
        # predict car keypoints
        res = predictor.images([left_image_path, right_image_path])
        pred_left, _, _ = next(res)
        pred_right, _, _ = next(res)

        # compute 3D points from matched annotations
        matches = hungarian_centroid_match(pred_left, pred_right)
        point3d_list = []
        for ann_left, ann_right in matches:
            p3d = car_keypoints3d(ann_left, ann_right, (fx, fy), args.baseline, c_left)
            point3d_list.append(p3d)
        bev2d_list = car_point3d_to_bev2d(point3d_list)

        # match general car model to keypoints
        _, model_points, _ = apollo_skeleton24(scale=args.car_scale)
        model_points2d = np.array(model_points)[:, :2]  # only x, z for 2D
        registered2d_list = register_car_model(bev2d_list, model_points2d, is_scaled=False,
                                               num_iter=args.ransac_iter, threshold=args.ransac_threshold)

        # annotate results
        image_left = cv2.imread(left_image_path)
        image_right = cv2.imread(right_image_path)
        annotate_left = car_annotate(image_left, pred_left)
        annotate_right = car_annotate(image_right, pred_right)
        # combine the two images
        combined_image = np.hstack((annotate_left, annotate_right))
        car_draw_matches(combined_image, matches, image_left.shape, image_right.shape)
        cv2.imwrite(os.path.join(frame_output_dir, 'car', 'annotated_car.png'), combined_image)

        # ========== Lane Line Part ========== #
        with torch.no_grad():
            _, left_ll_masks = get_mask(left_image_path, yolo_model, args, device, half)
            _, right_ll_masks = get_mask(right_image_path, yolo_model, args, device, half)

        left_ll_mask, right_ll_mask = left_ll_masks[0], right_ll_masks[0]

        # resize masks to original image size
        left_ll_mask = cv2.resize(left_ll_mask.astype(
            np.uint8), (image_left.shape[1], image_left.shape[0]), interpolation=cv2.INTER_NEAREST)
        right_ll_mask = cv2.resize(right_ll_mask.astype(
            np.uint8), (image_right.shape[1], image_right.shape[0]), interpolation=cv2.INTER_NEAREST)

        # save lane line masks
        cv2.imwrite(os.path.join(frame_output_dir, "lane", "left_lane_mask.png"),
                    (left_ll_mask * 255).astype(np.uint8))
        cv2.imwrite(os.path.join(frame_output_dir, "lane", "right_lane_mask.png"),
                    (right_ll_mask * 255).astype(np.uint8))

        # skeletonized sampling
        left_pts = sample_skeleton_points(left_ll_mask, args.num_samples)

        # save skeleton points
        skel_image = ll_draw_skeleton_points(image_left, left_pts)
        cv2.imwrite(os.path.join(frame_output_dir, "lane", "left_lane_skeleton.png"), skel_image)

        # compute depth map
        depth_map, disparity_map = compute_depth_sgbm(image_left, image_right, fx, args.baseline)

        # save disparity map & log-scaled depth map
        ll_disparity_map(disparity_map, save_path=os.path.join(frame_output_dir, 'lane', 'disparity_map.png'),
                         min_disp=args.min_disp, num_disp=args.num_disp)
        ll_log_depth_map(depth_map, save_path=os.path.join(frame_output_dir, 'lane', 'log_depth_map.png'))

        # compute 3D points from lane line points
        left_pts3d_list, valid_pts = ll_points3d(left_pts, depth_map, fx, c_left)

        # convert 3D points to BEV 2D points
        ll_bev2d_list = ll_point3d_to_bev2d(left_pts3d_list)

        # annotate results
        annotated_image = ll_annotate(image_left, left_pts3d_list, valid_pts)
        cv2.imwrite(os.path.join(frame_output_dir, 'lane', 'annotated_lane_line.png'), annotated_image)

        # clustering lane line points
        if args.clustering_method == 'dbscan':
            ll_clusters = group_ll_dbscan(ll_bev2d_list, eps=args.eps, min_samples=args.min_samples,
                                          z_max=args.z_max, z_weight=args.z_weight, use_z=args.use_z)
        elif args.clustering_method == 'kmeans':
            ll_clusters = group_ll_kmeans(ll_bev2d_list, n_clusters=args.n_clusters,
                                          z_max=args.z_max, z_weight=args.z_weight, use_z=args.use_z)

        # fit lane lines
        fitted_curves = fit_lane_lines(ll_clusters, fit_type=args.fit_type, line_conf=args.line_conf, scale=args.bev_scale,
                                       num_line_points=args.num_line_points)

        # plot bird's eye view
        if len(pred_left) > 0:
            bev(bev2d_list, pred_left[0].skeleton_m1, fitted_curves,
                save_path=os.path.join(frame_output_dir, 'bev.png'), scale=args.bev_scale)
            bev(registered2d_list, pred_left[0].skeleton_m1, fitted_curves,
                save_path=os.path.join(frame_output_dir, 'bev_registered.png'), scale=args.bev_scale)
        else:
            # Handle case where no detections are found
            bev(bev2d_list, None, fitted_curves,
                save_path=os.path.join(frame_output_dir, 'bev.png'), scale=args.bev_scale)
            bev(registered2d_list, None, fitted_curves,
                save_path=os.path.join(frame_output_dir, 'bev_registered.png'), scale=args.bev_scale)

        return True

    except Exception as e:
        LOG.error(f"Error processing {frame_name}: {str(e)}")
        return False


def main():
    # parse command line arguments
    args = cli()

    # Validate input directory structure
    left_dir = os.path.join(args.data_dir, 'left')
    right_dir = os.path.join(args.data_dir, 'right')

    if not os.path.exists(left_dir):
        raise RuntimeError(f"Left image directory not found: {left_dir}")
    if not os.path.exists(right_dir):
        raise RuntimeError(f"Right image directory not found: {right_dir}")

    # Create main output directory
    os.makedirs(args.output, exist_ok=True)

    # Get all image pairs
    image_pairs = get_image_pairs(args.data_dir)
    if not image_pairs:
        raise RuntimeError("No matching image pairs found")

    LOG.info(f"Found {len(image_pairs)} image pairs to process")

    # Set up models (initialize once for all frames)
    # pifpaf model pipeline
    predictor = Predictor(
        visualize_image=False,
        visualize_processed_image=args.debug,
    )

    # YOLOPv2 model
    device = select_device(args.device.type)
    half = device.type != 'cpu'
    yolo_model = torch.jit.load(args.ckpt).to(device)
    if half:
        yolo_model.half()
    yolo_model.eval()

    # Process each image pair
    successful_count = 0
    failed_count = 0

    for left_path, right_path, frame_name in tqdm(image_pairs, desc="Processing frames"):
        LOG.info(f"Processing frame: {frame_name}")

        success = process_single_pair(left_path, right_path, frame_name, args,
                                      predictor, yolo_model, device, half)

        if success:
            successful_count += 1
        else:
            failed_count += 1

    LOG.info(f"Processing complete. Success: {successful_count}, Failed: {failed_count}")


if __name__ == '__main__':
    main()
