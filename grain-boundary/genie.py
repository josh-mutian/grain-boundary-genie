import sys
import copy
import numpy as np
from structure import Structure
import geometry as geom

def gb_genie(struct, orien_1, orien_2, twist_agl, trans_vec):
    view_agls = np.array([[1, 0, 0], [1, 1, 0], [2, 1, 0], [1, 1, 1], [2, 1, 1]]).astype(float)
    view_agl, _ = geom.mutual_view_angle(orien_1, orien_2, view_agls, np.deg2rad(10))
    trans_1 = np.vstack([view_agl, np.cross(view_agl, orien_1), orien_1])
    trans_1 = np.apply_along_axis(geom.normalize_vector, 1, trans_1)
    trans_2 = np.array([view_agl, np.cross(view_agl, orien_2), orien_2])
    trans_2 = np.apply_along_axis(geom.normalize_vector, 1, trans_2)
    struct_1 = struct
    struct_2 = copy.deepcopy(struct_1)

    struct_1.transform(trans_1)
    struct_1.to_vasp('1_transform')
    struct_2.transform(trans_2)
    struct_2.to_vasp('2_transform')
    # raise NotImplementedError('gb_genie() not implemented.')
    return

def main(argv):
    struct = Structure.from_vasp(argv[1])
    gb_genie(struct, np.array([1, 0.601942, 0.0296519]), np.array([1, 0.730728, 0.220564]), 0.0, None)
    
if __name__ == '__main__':
    main(sys.argv)
