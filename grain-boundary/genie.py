import sys
import copy
import numpy as np
from structure import Structure
import geometry as geom

def gb_genie(struct, orien_1, orien_2, twist_agl, trans_vec):
    trans_1 = np.dot(geom.rotation_angle_matrix(np.array([0., 0., 1.]), 
                                                twist_agl),
                     geom.get_rotatino_matrix(orien_1, np.array([0., 0., 1.])))
    trans_2 = geom.get_rotatino_matrix(orien_2, np.array([0., 0., 1.]))
    box_1 = np.transpose(np.dot(trans_1, np.transpose(struct.coordinates)))
    box_2 = np.transpose(np.dot(trans_2, np.transpose(struct.coordinates)))
    box_1 = geom.rebase_coord_sys(box_1) + trans_vec
    box_2 = geom.rebase_coord_sys(box_2)

def find_coincident_points(box_1, box_2, max_int, tol):
    res = []
    for i in range(tol):
        for j in range(tol):
            for k in range(tol):
                vec = np.dot(np.array([i, j, k]), box_1)
                nearest_int_mult = np.rint(np.dot(vec, np.linalg.inv(box_2)))
                dist = np.linalg.norm(np.dot(nearest_int_mult, box_2) - vec)
                if dist <= tol:
                    res.append(vec)
    res = np.array(np)
    res = res[np.argsort(np.apply_along_axis(np.linalg.norm, 1, res))]
    return res


def main(argv):
    struct = Structure.from_vasp(argv[1])
    gb_genie(struct, np.array([1, 0.601942, 0.0296519]), np.array([1, 0.730728, 0.220564]), 0.0, None)
    
if __name__ == '__main__':
    main(sys.argv)
