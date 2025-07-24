import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from .car.matching import get_centroid

def bev(bev2d_list, skeleton, ll_clusters_dict, save_path='bev.png', figsize=(8, 8), point_size=4, scale=1800):
    """
    Generate a bird's eye view from 3D points.
    """
    _, ax = plt.subplots(figsize=figsize)
    car_cmap = plt.colormaps['jet'].resampled(len(skeleton))
    ll_cmap = plt.cm.get_cmap('tab10', len(ll_clusters_dict))

    # Plot lane line
    for i, (label, points) in enumerate(ll_clusters_dict.items()):
        if len(points) == 0:
            continue
        x = points[1]
        z = points[0]
        ax.scatter(x, z, s=point_size, color=ll_cmap(i), label=f'Lane {label}', zorder=2)
    ax.legend(loc='upper right')

    for points in bev2d_list:
        # Draw skeleton
        for idx, (i, j) in enumerate(skeleton):
            if i < len(points) and j < len(points):
                pi = points[i]
                pj = points[j]
                if pi is not None and pj is not None:
                    xi, zi = pi[0], pi[1]
                    xj, zj = pj[0], pj[1]
                    ax.plot([xi, xj], [zi, zj], color=car_cmap(idx), linewidth=2)
        # Draw keypoints
        for pt in points:
            if pt is not None:
                x, z = pt
                ax.scatter(x, z, s=point_size, c='black', zorder=3)

    ax.set_xlabel("X (cm)")
    ax.set_ylabel("Z (cm)")
    ax.set_title("Bird's Eye View")
    ax.grid(True)
    ax.set_xlim(-scale/2, scale/2)
    ax.set_ylim(0, scale)

    # os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved BEV image to {save_path}")

def car_annotate(image, pred):
    """
    Annotate the image with car keypoints, skeletons, and bounding boxes.
    """
    annotated_image = image.copy()
    for ann in pred:
        # extract data
        num_bones = len(ann.skeleton_m1)
        keypoints = ann.data[:, :3]  # x, y, confidence
        x, y, w, h = ann.bbox()
        # Draw bounding box
        cv2.rectangle(annotated_image, (int(x), int(y)), (int(x + w), int(y + h)), color=(0, 255, 0), thickness=2)
        # Draw bones
        for idx, (joint_a, joint_b) in enumerate(ann.skeleton_m1):
            if keypoints[joint_a][2] > 0.0 and keypoints[joint_b][2] > 0.0:
                pt1 = tuple(int(v) for v in keypoints[joint_a][:2])
                pt2 = tuple(int(v) for v in keypoints[joint_b][:2])

                # Map index to color (without normalization)
                color_idx = int(255 * idx / max(num_bones - 1, 1))
                color = cv2.applyColorMap(np.array([[color_idx]], dtype=np.uint8), cv2.COLORMAP_JET)[0, 0].tolist()

                cv2.line(annotated_image, pt1, pt2, color=color, thickness=3)
        # Draw keypoints
        for x, y, conf in keypoints:
            if conf > 0.0:
                cv2.circle(annotated_image, (int(x), int(y)), 3, color=(255, 0, 0), thickness=-1)
    return annotated_image

def car_draw_matches(combined_image, matches, shape_left, shape_right):
    """
    Draw matches between left and right images.
    """
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
        
def ll_draw_skeleton_points(image, pts, color=(0, 255, 255), radius=2):
    """
    Draw skeleton points on the image.
    """
    skel_image = image.copy()
    for x, y in pts.astype(int):
        if 0 <= x < skel_image.shape[1] and 0 <= y < skel_image.shape[0]:
            cv2.circle(skel_image, (x, y), radius, color, -1)
    
    return skel_image

def ll_disparity_map(disparity, save_path='disparity_map.png', min_disp=0, num_disp=128):
    """
    Visualize the disparity map.
    """
    disp_norm = np.nan_to_num(disparity, nan=0.0)
    disp_norm[disp_norm < min_disp] = 0
    disp_vis = ((disp_norm - min_disp) / num_disp * 255).clip(0, 255).astype(np.uint8)
    cv2.imwrite(save_path, disp_vis)

def ll_log_depth_map(depth_map, save_path='log_depth_map.png'):
    """
    Visualize the log-scaled depth map.
    """
    log_depth = np.log(depth_map + 1)
    log_depth = (log_depth / np.nanmax(log_depth) * 255).astype(np.uint8)
    cv2.imwrite(save_path, log_depth)

def ll_annotate(image, pts3d_list, valid_pts):
    """
    Annotate the image lane line points and depth.
    """
    annotated_image = image.copy()
    pts_3d = np.array(pts3d_list)
    valid_pts = np.array(valid_pts)
    if pts_3d.size == 0:
        return annotated_image

    for (x, y), z in zip(valid_pts, pts_3d[:, 2]):
        cv2.putText(annotated_image, f"{(z/100):.2f}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
        cv2.circle(annotated_image, (x, y), 3, (0, 255, 0), -1)

    return annotated_image