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
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt


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


def out_name(arg, in_name, default_extension):
    """Determine an output name from args, input name and extension.

    arg can be:
    - none: return none (e.g. show image but don't store it)
    - True: activate this output and determine a default name
    - string:
        - not a directory: use this as the output file name
        - is a directory: use directory name and input name to form an output
    """
    if arg is None:
        return None

    if arg is True:
        return in_name + default_extension

    if os.path.isdir(arg):
        return os.path.join(
            arg,
            os.path.basename(in_name)
        ) + default_extension

    return arg

def load_projection_matrices(calib_file):
    P_rect_02 = None
    P_rect_03 = None
    with open(calib_file, 'r') as f:
        for line in f:
            if line.startswith('P_rect_02'):
                P_rect_02 = np.array([float(x) for x in line.split()[1:]]).reshape(3, 4)
            elif line.startswith('P_rect_03'):
                P_rect_03 = np.array([float(x) for x in line.split()[1:]]).reshape(3, 4)
    return P_rect_02, P_rect_03

def annotate_image(image, pred):
    """Annotate the image with keypoints, skeletons, and bounding boxes."""
    for ann in pred:
        # extract data
        num_bones = len(ann.skeleton_m1)
        keypoints = ann.data[:, :3]  # x, y, confidence
        x, y, w, h = ann.bbox()
        # Draw bounding box
        cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), color=(0, 255, 0), thickness=2)
        # Draw bones
        for idx, (joint_a, joint_b) in enumerate(ann.skeleton_m1):
            if keypoints[joint_a][2] > 0.0 and keypoints[joint_b][2] > 0.0:
                pt1 = tuple(int(v) for v in keypoints[joint_a][:2])
                pt2 = tuple(int(v) for v in keypoints[joint_b][:2])

                # Map index to color (without normalization)
                color_idx = int(255 * idx / max(num_bones - 1, 1))
                color = cv2.applyColorMap(np.array([[color_idx]], dtype=np.uint8), cv2.COLORMAP_JET)[0, 0].tolist()

                cv2.line(image, pt1, pt2, color=color, thickness=3)
        # Draw keypoints
        for x, y, conf in keypoints:
            if conf > 0.0:
                cv2.circle(image, (int(x), int(y)), 3, color=(255, 0, 0), thickness=-1)
    return image

def get_centroid(ann):
    """Get the centroid from a annotation."""
    x, y, w, h = ann.bbox()
    return (x + w / 2, y + h / 2)

def hungarian_centroid_match(pred_left, pred_right):
    """
    Match keypoints between left and right predictions using the Hungarian algorithm.
    Note: The constraint threshold may need to be changed into a percentage of the image size rather than pixel for better generalization.
    """
    # get centroids of the bounding boxes
    left_centroids = np.array([get_centroid(ann) for ann in pred_left])
    right_centroids = np.array([get_centroid(ann) for ann in pred_right])
    # if no centroids, return empty list
    if len(left_centroids) == 0 or len(right_centroids) == 0:
        return []
    # Hungarian algorithm to find the best matches
    cost_matrix = np.linalg.norm(left_centroids[:, np.newaxis] - right_centroids, axis=2)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    # filter matches based on a threshold
    matches = []
    for i, j in zip(row_ind, col_ind):
        x_threshold = np.abs(left_centroids[i][0] - right_centroids[j][0]) < 300
        # y coordinate should follow the constrint of rectified stereo pair, thus stricter
        y_threshold = np.abs(left_centroids[i][1] - right_centroids[j][1]) < 25
        if x_threshold and y_threshold:
            matches.append((pred_left[i], pred_right[j]))
    return matches

def draw_matches(combined_image, matches, shape_left, shape_right):
    """Draw matches between left and right images."""
    for ann_left, ann_right in matches:
        centroid_left = get_centroid(ann_left)
        centroid_right = get_centroid(ann_right)
        # draw centroids
        cv2.circle(combined_image, (int(centroid_left[0]), int(centroid_left[1])), 5, color=(0, 0, 255), thickness=5)
        cv2.circle(combined_image, (int(centroid_right[0] + shape_left[1]), int(centroid_right[1])), 5, color=(0, 0, 255), thickness=5)
        # draw line connecting the two annotations
        cv2.line(combined_image,
                 (int(centroid_left[0]), int(centroid_left[1])),
                 (int(centroid_right[0] + shape_left[1]), int(centroid_right[1])),
                 color=(0, 0, 255), thickness=5)
        
def keypoints3d(ann_left, ann_right, focal_length:tuple, baseline:float, centriod_left:tuple):
    """Estimate depth from matched objects."""
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

def bev(points_3d_list, skeleton, save_path='bev.png', figsize=(8, 8), point_size=4):
    """Generate a bird's eye view from 3D points."""
    fig, ax = plt.subplots(figsize=figsize)
    cmap = plt.colormaps['jet'].resampled(len(skeleton))

    for points in points_3d_list:
        # Draw skeleton
        for idx, (i, j) in enumerate(skeleton):
            if i < len(points) and j < len(points):
                pi = points[i]
                pj = points[j]
                if pi is not None and pj is not None:
                    xi, zi = pi[0], pi[2]
                    xj, zj = pj[0], pj[2]
                    ax.plot([xi, xj], [zi, zj], color=cmap(idx), linewidth=2)
        # Draw keypoints
        for pt in points:
            if pt is not None:
                x, _, z = pt
                ax.scatter(x, z, s=point_size, c='black', zorder=3)

    ax.set_xlabel("X (cm)")
    ax.set_ylabel("Z (cm)")
    ax.set_title("Bird's Eye View")
    ax.grid(True)
    ax.set_xlim(-900, 900)
    ax.set_ylim(0, 1800)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved BEV image to {save_path}")

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
    pred_left, _, meta_left = next(res)
    pred_right, _, meta_right = next(res)

    # compute depth from disparity
    matches = hungarian_centroid_match(pred_left, pred_right)
    points3d = []
    for ann_left, ann_right in matches:
        p3d = keypoints3d(ann_left, ann_right, (fx, fy), BASELINE, c_left)
        points3d.append(p3d)

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
    bev(points3d, pred_left[0].skeleton_m1, save_path=os.path.join(args.output, 'bev.png'))


if __name__ == '__main__':
    main()
