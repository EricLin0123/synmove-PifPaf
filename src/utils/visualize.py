import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from .matching import get_centroid

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