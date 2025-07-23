import numpy as np
from scipy.optimize import linear_sum_assignment

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