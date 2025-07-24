import numpy as np
from sklearn.cluster import DBSCAN, KMeans

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