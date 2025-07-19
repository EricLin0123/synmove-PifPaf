from openpifpaf.plugins.apollocar3d import constants
import matplotlib.pyplot as plt

def apollo_skeleton24():
    """
    Get the keypoints and skeleton for ApolloCar3D with 24 keypoints.
    """
    names = constants.CAR_KEYPOINTS_24
    keypoints = constants.CAR_POSE_24
    skeleton = [(i-1, j-1) for (i, j) in constants.CAR_SKELETON_24]
    # post process keypoints
    keypoints = keypoints.copy()
    keypoints[:, 1], keypoints[:, 2] = -keypoints[:, 2], keypoints[:, 1]
    keypoints[:, 0] *= -1  # Invert X
    # scale keypoints to meters
    keypoints *= 40
    return names, keypoints, skeleton


if __name__ == "__main__":
    names, keypoints, skeleton = apollo_skeleton24()
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # ax.view_init(elev=0, azim=-90)
    ax.scatter(keypoints[:, 0], keypoints[:, 1], keypoints[:, 2], c='b', s=30)
    cmap = plt.colormaps['jet'].resampled(len(skeleton))
    for idx, (i, j) in enumerate(skeleton):
        x = [keypoints[i, 0], keypoints[j, 0]]
        y = [keypoints[i, 1], keypoints[j, 1]]
        z = [keypoints[i, 2], keypoints[j, 2]]
        ax.plot(x, y, z, color=cmap(idx), linewidth=2)
    for name, (x, y, z) in zip(names, keypoints):
        ax.text(x, y, z, name, fontsize=8, color='black')
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    # ax.set_xlim([0, 10])
    # ax.set_ylim([-10, 0])
    # ax.set_zlim([0, 10])
    ax.set_title("ApolloCar3D Keypoints 24")
    plt.savefig("car_keypoints_3d.png")