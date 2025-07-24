import numpy as np

def load_projection_matrices(calib_file):
    """
    Load the projection matrices from the calibration file.
    """
    P_rect_02 = None
    P_rect_03 = None
    with open(calib_file, 'r') as f:
        for line in f:
            if line.startswith('P_rect_02'):
                P_rect_02 = np.array([float(x) for x in line.split()[1:]]).reshape(3, 4)
            elif line.startswith('P_rect_03'):
                P_rect_03 = np.array([float(x) for x in line.split()[1:]]).reshape(3, 4)
    return P_rect_02, P_rect_03

def linear_line(z, a, b):
    """
    Linear function for fitting lane lines.
    """
    return a * z + b

def quadratic_line(z, a, b, c):
    """
    Quadratic function for fitting lane lines.
    """
    return a * z**2 + b * z + c