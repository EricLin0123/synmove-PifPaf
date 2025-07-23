import numpy as np

def noncoplanar_registration(A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform non-coplanar point cloud registration of source image to target image.
    For 2 non-coplanar anisotropic scaled point clouds,
    this function computes the transform (R, S, t) such that:
    A = R * S * B + t.
    Parameters:
    - A: Source point cloud (dimxN)
    - B: Target point cloud (dimxN) matched with A by indices.
    Returns:
    - R: Rotation matrix (dimxdim)
    - S: Scaling matrix (dimxdim)
    - t: Translation vector (dimx1)
    """
    # input checks
    if A.shape[0] != B.shape[0]:
        raise ValueError("Matched point clouds must have the same dimension.")
    if A.shape[1] != B.shape[1]:
        raise ValueError("Matched point clouds must have the same number of points.")
    # calculate centroids of the point clouds
    centroid_A = np.mean(A, axis=1, keepdims=True)
    centroid_B = np.mean(B, axis=1, keepdims=True)
    # center the point clouds
    A_centered = A - centroid_A
    B_centered = B - centroid_B
    # stable SVD computation
    H = A_centered @ B_centered.T # the first term AB^T
    P = B_centered @ B_centered.T # the second term BB^T
    UH, SH, VhH = np.linalg.svd(H, full_matrices=False)
    UP, SP, VhP = np.linalg.svd(P, full_matrices=False)
    Hd = UH @ np.diag(SH) @ VhH
    Pd = UP @ np.diag(SP) @ VhP
    U, _, Vh = np.linalg.svd(Hd @ np.linalg.pinv(Pd), full_matrices=False) # SVD of AB^T * (BB^T)^-1
    # obtain rotation matrix R
    D = np.eye(U.shape[0])
    if np.linalg.det(U) * np.linalg.det(Vh) < 0:
        D[-1, -1] = -1
    R = U @ D @ Vh
    # obtain scaling matrix S
    dim = A.shape[0]
    s = []
    for i in range(dim):
        Lamb = np.zeros((dim, dim))
        Lamb[i, i] = 1
        num = np.trace(A_centered.T @ R @ Lamb @ B_centered)
        den = np.trace(B_centered.T @ Lamb @ B_centered)
        s.append(num / (den + 1e-8)) # si = trace(A^T R L_i B) / trace(B^T L_i B)
    S = np.diag(s)
    # obtain translation vector t
    t = (centroid_A - R @ S @ centroid_B)

    return R, S, t

def unscaled_registration(A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform unscaled point cloud registration of source image to target image.
    For 2 point clouds, this function computes the transform (R, t) such that:
    A = R * B + t.
    Parameters:
    - A: Source point cloud (dimxN)
    - B: Target point cloud (dimxN) matched with A by indices.
    Returns:
    - R: Rotation matrix (dimxdim)
    - S: Identity matrix (dimxdim, identity for unscaled registration)
    - t: Translation vector (dimx1)
    """
    # input checks
    if A.shape[0] != B.shape[0]:
        raise ValueError("Matched point clouds must have the same dimension.")
    if A.shape[1] != B.shape[1]:
        raise ValueError("Matched point clouds must have the same number of points.")
    
    # calculate centroids of the point clouds
    centroid_A = np.mean(A, axis=1, keepdims=True)
    centroid_B = np.mean(B, axis=1, keepdims=True)
    
    # center the point clouds
    A_centered = A - centroid_A
    B_centered = B - centroid_B
    
    # stable SVD computation
    H = A_centered @ B_centered.T
    U, _, Vh = np.linalg.svd(H, full_matrices=False)
    
    # obtain rotation matrix R
    D = np.eye(U.shape[0])
    if np.linalg.det(U) * np.linalg.det(Vh) < 0:
        D[-1, -1] = -1
    R = U @ D @ Vh
    
    # obtain translation vector t
    t = (centroid_A - R @ centroid_B)

    # scale is identity for unscaled registration
    S = np.eye(A.shape[0])

    return R, S, t

def car_registration_ransac(point3d, model_points, num_iter:int, is_scaled:bool=False, threshold:float=100):
    '''
    Register a car model with outlier filtering.
    '''
    # extract valid points
    A_all, B_all = [], []
    for idx, pt in enumerate(point3d):
        if pt is not None: # valid 3D point
            A_all.append(list(pt))
            B_all.append(model_points[idx])
    A_all = np.array(A_all).T
    B_all = np.array(B_all).T
    # check if we have enough points
    if A_all.shape[1] < 3: # no ransac if not enough points
        print("Not enough points for registration, using all points without ransac.")
        print(f"Number of points: {A_all.shape[1]}")
        if is_scaled:
            R, S, t = noncoplanar_registration(A_all, B_all)
        else:
            R, S, t = unscaled_registration(A_all, B_all)
        registered_points = (R @ S @ model_points.T + t).T
        return registered_points.tolist()
    # ransac
    best_inliers = []
    for _ in range(num_iter):
        indecies = np.random.choice(A_all.shape[1], size=3, replace=False)
        A = A_all[:, indecies]
        B = B_all[:, indecies]
        if is_scaled:
            R, S, t = noncoplanar_registration(A, B)
        else:
            R, S, t = unscaled_registration(A, B)
        A_pred = R @ S @ B_all + t
        # calculate error
        errors = np.linalg.norm(A_pred - A_all, axis=0)
        inliers = np.where(errors < threshold)[0]  # threshold for inliers
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
    # best inlier
    print(f"Found {len(best_inliers)} inliers out of {A_all.shape[1]} points.")
    if len(best_inliers) > 3:
        A_best = A_all[:, best_inliers]
        B_best = B_all[:, best_inliers]
        if is_scaled:
            R, S, t = noncoplanar_registration(A_best, B_best)
        else:
            R, S, t = unscaled_registration(A_best, B_best)
    else:
        if is_scaled:
            R, S, t = noncoplanar_registration(A_all, B_all)
        else:
            R, S, t = unscaled_registration(A_all, B_all)

    registered_points = (R @ S @ model_points.T + t).T
    return registered_points.tolist()

def register_car_model(point3d_list:list[list], model_points:np.ndarray, is_scaled:bool=False,
                        num_iter:int=100, threshold:float=50) -> list[list]:
    '''
    Register a list of 3D points to a car model using point cloud registration.
    '''
    assert len(point3d_list[0]) == len(model_points), "Number of points in 3D list must match model points."
    registered_point_list = []
    for points in point3d_list:
        registered_points = car_registration_ransac(points, model_points, num_iter=num_iter, is_scaled=is_scaled, threshold=threshold)
        registered_point_list.append(registered_points)
    return registered_point_list