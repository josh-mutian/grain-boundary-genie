import numpy as np


def get_rotation_matrix(a, b):
    '''
    a, b are angles represented by a 1*3 arrays.
    a -> b
    ref: http://math.stackexchange.com/questions/293116/rotating-one-3d-vector-to-another
    '''
    a = np.array(a)
    b = np.array(b)
    if (len(a) != 3 or len(b) != 3):
        raise ValueError("a and b must be of length 3.")
    x = np.cross(a, b)
    x = x / np.linalg.norm(x)

    cos_theta = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    theta = np.arccos(cos_theta)
    A = np.array([
        [0, -x[2],  x[1]],
        [x[2],     0, -x[0]],
        [-x[1],  x[0],     0]
    ])  # skew-symmetric cross-product of v
    r = np.identity(3) + np.sin(theta) * A + (1 - cos_theta) * np.dot(A, A)
    return r
