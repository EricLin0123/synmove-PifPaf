import numpy as np
from sklearn.cluster import DBSCAN, KMeans
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit
from ..common import linear_line, quadratic_line

def group_ll_dbscan(pts_2d, eps=100, min_samples=20, z_max=3000, z_weight=0.2, use_z=False):
    """
    Group lane line points using DBSCAN clustering.
    """
    if len(pts_2d) == 0:
        return {}

    pts_2d = np.array(pts_2d)

    mask = pts_2d[:, 1] <= z_max
    pts_2d = pts_2d[mask]

    x = pts_2d[:, 0]
    z = pts_2d[:, 1]

    if use_z:
        features = np.stack([x, z * z_weight], axis=1)
    else:
        features = x.reshape(-1, 1)

    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(features)

    clusters = {}
    for label in np.unique(labels):
        clusters[label] = pts_2d[labels == label]

    return clusters

def group_ll_kmeans(pts_2d, n_clusters=4, z_max=3000, z_weight=0.2, use_z=False):
    """
    Group lane line points using KMeans clustering.
    """
    if len(pts_2d) == 0:
        return {}

    pts_2d = np.array(pts_2d)

    mask = pts_2d[:, 1] <= z_max
    pts_2d = pts_2d[mask]

    x = pts_2d[:, 0]
    z = pts_2d[:, 1]

    if use_z:
        features = np.stack([x, z * z_weight], axis=1)
    else:
        features = x.reshape(-1, 1)

    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init='auto')
    labels = kmeans.fit_predict(features)

    clusters = {}
    for label in np.unique(labels):
        clusters[label] = pts_2d[labels == label]

    return clusters

def fit_lane_lines(clusters, fit_type='linear', line_conf=500, scale=1800, num_line_points=100):
    """ 
    Fit lane lines to clustered points.
    """
    fitted_curves = {}
    reference_popt = None
    missing_ids = []

    for cluster_id, points in clusters.items():
        if cluster_id == -1 or len(points) < 3:
            continue

        # Extract and sort by z for stability
        points = points[np.argsort(points[:, 1])]
        x = points[:, 0]
        z = points[:, 1]

        try:
            if (fit_type == 'linear'):
                # fit linear line
                popt, _ = curve_fit(linear_line, z, x)
                fitted_x = linear_line(z, *popt)
            elif (fit_type == 'quadratic'):
                # fit quadratic curve
                popt, _ = curve_fit(quadratic_line, z, x)
                fitted_x = quadratic_line(z, *popt)

            # filter lines that points are too close to each other (i.e. not complete lane lines)
            if np.abs(fitted_x[-1] - fitted_x[0]) < line_conf and np.abs(z[-1] - z[0]) < line_conf:
                missing_ids.append(cluster_id)
                fitted_curves[cluster_id] = None
                print(f"[WARN] Cluster {cluster_id} is too short or not complete.")
            else:
                # extend fitted curve to the scale
                z = np.linspace(0, scale, num=num_line_points)
                if fit_type == 'linear':
                    fitted_x = linear_line(z, *popt)
                elif fit_type == 'quadratic':
                    fitted_x = quadratic_line(z, *popt) 
                fitted_curves[cluster_id] = (z, fitted_x)

                # set reference parameters (except the constant) for the first valid cluster
                if reference_popt is None:
                    reference_popt = popt[:-1]  # exclude the constant term

        except RuntimeError:
            missing_ids.append(cluster_id)
            fitted_curves[cluster_id] = None
            print(f"[WARN] Could not fit curve for cluster {cluster_id}")

    # recover missing lane lines using the line coefficient of reference lane line
    # here we assume that the lane lines are parallel
    if reference_popt is not None:
        for cluster_id in missing_ids:
            if cluster_id not in fitted_curves:
                continue
            z = np.linspace(0, scale, num=num_line_points)
            # use mean point of the cluster to find the constant term
            mean_x = np.mean(clusters[cluster_id][:, 0])
            mean_z = np.mean(clusters[cluster_id][:, 1])
            if fit_type == 'linear':
                constant = mean_x - (reference_popt[0] * mean_z)
                popt = np.append(reference_popt, constant)
                fitted_x = linear_line(z, *popt) 
            elif fit_type == 'quadratic':
                constant = mean_x - (reference_popt[0] * mean_z**2 + reference_popt[1] * mean_z)
                fitted_curves[cluster_id] = (mean_z, quadratic_line(mean_z, *reference_popt, constant))
                popt = np.append(reference_popt, constant)
                fitted_x = quadratic_line(z, *popt)
            fitted_curves[cluster_id] = (z, fitted_x)
            
    return fitted_curves