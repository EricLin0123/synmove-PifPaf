"""Predict poses for given images."""
import argparse
import glob
import logging
import os
import torch
import cv2
import numpy as np
from openpifpaf import decoder, logger, network, show, visualizer, __version__
from openpifpaf.predictor import Predictor
from utils.visualize import annotate_image, draw_matches, bev, full_bev # for all the plots
from utils.matching import hungarian_centroid_match # for matching vehicles between left and right images
from utils.registration import register_car_model # for matching keypoints with full car model
from utils.apollo_skeleton import apollo_skeleton24 # for car keypoints and skeleton

from laneline import get_ll_points3d

LOG = logging.getLogger(__name__)
BASELINE = 45 # cm

def cli():
    parser = argparse.ArgumentParser(
        prog='python3 -m openpifpaf.predict',
        usage='%(prog)s [options] images',
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

    parser.add_argument('images',
                        help='input images directory', default="data")
    parser.add_argument('--glob',
                        help='glob expression for input images (for many images)')
    parser.add_argument('--output', type=str, default="output",
                        help='output directory')
    parser.add_argument('--disable-cuda', action='store_true',
                        help='disable CUDA')
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

    # glob
    if args.glob:
        args.images += glob.glob(args.glob)
    if not args.images:
        raise RuntimeError("no image files given")

    return args

def load_projection_matrices(calib_file):
    '''Load the projection matrices from the calibration file.'''
    P_rect_02 = None
    P_rect_03 = None
    with open(calib_file, 'r') as f:
        for line in f:
            if line.startswith('P_rect_02'):
                P_rect_02 = np.array([float(x) for x in line.split()[1:]]).reshape(3, 4)
            elif line.startswith('P_rect_03'):
                P_rect_03 = np.array([float(x) for x in line.split()[1:]]).reshape(3, 4)
    return P_rect_02, P_rect_03

def keypoints3d(ann_left, ann_right, focal_length:tuple, baseline:float, centriod_left:tuple):
    '''Estimate 3D position from matched objects.'''
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

def point3d_to_bev2d(point3d_list):
    """Convert 3D points to 2D BEV points."""
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

def main():
    # parse command line arguments
    args = cli()
    left_image_path = os.path.join(args.images, 'left.png')
    right_image_path = os.path.join(args.images, 'right.png')
    calib_file_path = os.path.join(args.images, 'calib.txt')
    os.makedirs(args.output, exist_ok=True)

    # load projection matrices
    P_left, P_right = load_projection_matrices(calib_file_path)
    fx = P_left[0, 0]  # focal length
    fy = P_left[1, 1]  # focal length
    c_left = [P_left[0, 2], P_left[1, 2]]  # principal point left

    # setup model pipeline
    predictor = Predictor(
        visualize_image=False,
        visualize_processed_image=args.debug,
    )

    # predict car keypoints
    res = predictor.images([left_image_path, right_image_path])
    pred_left, _, _ = next(res)
    pred_right, _, _ = next(res)

    # compute 3D points from matched annotations
    matches = hungarian_centroid_match(pred_left, pred_right)
    point3d_list = []
    for ann_left, ann_right in matches:
        p3d = keypoints3d(ann_left, ann_right, (fx, fy), BASELINE, c_left)
        point3d_list.append(p3d)
    bev2d_list = point3d_to_bev2d(point3d_list)

    # match general car model to keypoints
    _, model_points, _ = apollo_skeleton24()
    model_points2d = np.array(model_points)[:, :2]  # only x, z for 2D
    registered2d_list = register_car_model(bev2d_list, model_points2d, is_scaled=False)

    # annotate results
    image_left = cv2.imread(left_image_path)
    image_right = cv2.imread(right_image_path)
    annotate_left = annotate_image(image_left, pred_left)
    annotate_right = annotate_image(image_right, pred_right)
    # combine the two images
    combined_image = np.hstack((annotate_left, annotate_right))
    draw_matches(combined_image, matches, image_left.shape, image_right.shape)
    cv2.imwrite(os.path.join(args.output, 'annotation.png'), combined_image)

    # plot bird's eye view
    bev(bev2d_list, pred_left[0].skeleton_m1, save_path=os.path.join(args.output, 'bev.png'))
    bev(registered2d_list, pred_left[0].skeleton_m1, save_path=os.path.join(args.output, 'bev_registered.png'))

    # ==== Adding Lane Line ==== #
    ll_points3d = get_ll_points3d(left_image_path, right_image_path, P_left, P_right, fx, BASELINE, c_left)
    ll_bev2d = []
    for pt in ll_points3d:
        x, _, z = pt
        ll_bev2d.append((x, z))

    full_bev(bev2d_list, pred_left[0].skeleton_m1, ll_bev2d, save_path=os.path.join(args.output, 'full-bev.png'))
    full_bev(registered2d_list, pred_left[0].skeleton_m1, ll_bev2d, save_path=os.path.join(args.output, 'full-bev_registered.png'))


if __name__ == '__main__':
    main()
