import numpy as np
from skimage.morphology import skeletonize

def sample_skeleton_points(mask, num_points=300):
    """
    Sample points from the skeleton of a binary mask.
    """
    skeleton = skeletonize(mask > 0)
    coords = np.column_stack(np.where(skeleton))  # (y, x)
    if len(coords) == 0:
        return np.empty((0, 2), dtype=np.float32)
    if len(coords) > num_points:
        idxs = np.linspace(0, len(coords) - 1, num_points, dtype=int)
        coords = coords[idxs]
    return coords[:, ::-1].astype(np.float32)  # return (x, y)