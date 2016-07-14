import numpy as np

def get_rotation_matrix(a, b):
    '''
    a, b are angles represented by a 1*3 numpy array.
    a -> b
    ref: http://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d
    '''
    if (len(a) != 3 or len(b) != 3):
        raise ValueError("a and b must be of length 3.")
    v = np.cross(a, b)
    s = np.linalg.norm(v)
    c = np.dot(a, b)
    sscp = [
        [    0, -v[2],  v[1]],
        [ v[2],     0, -v[0]],
        [-v[1],  v[0],     0]
    ] # skew-symmetric cross-product of v
    r = np.identity(3) + sscp + np.dot(sscp, sscp) * (1 - c) / (s ** 2)