"""Routines to remove collision within a Structure object.
"""
import numpy as np
import geometry as geom


def apart_by_safe_distance(min_dist_dict, atom_1, atom_2):
    """Looks up the minimum distance dictionary and decide whether two atoms
        are apart by safe distance by two atoms.

    Args:
        min_dist_dict (dict): A dictionary where the key is tuple of atom type
            names and value is the minimum distance in angstrom.
        atom_1 (nparray): A record array representing an atom.
        atom_2 (nparray): A record array representing an atom.

    Returns:
        bool: True if distance is safe, False otherwise.
    """
    dist = np.linalg.norm(atom_1['position'] - atom_2['position'])
    try:
        min_dist = min_dist_dict[(atom_1['element'], atom_2['element'])]
    except KeyError, e:
        try:
            min_dist = min_dist_dict[(atom_2['element'], atom_1['element'])]
        except KeyError, e:
            return True
        else:
            return dist >= min_dist
    else:
        return dist >= min_dist


def is_safe_distance(min_dist_dict, dist, ele_1, ele_2):
    """Looks up the minimum distance dictionary and decide whether two atoms
        are apart by safe distance by distance and atom element names.

    Args:
        min_dist_dict (dict): A dictionary where the key is tuple of atom type
            names and value is the minimum distance in angstrom.
        dist (float): Distance between atoms.
        ele_1 (str): Element name of one atom.
        ele_2 (TYPE): Element name of another atom.

    Returns:
        bool: True if distance is safe, False otherwise.
    """
    try:
        min_dist = min_dist_dict[(ele_1, ele_2)]
    except KeyError, e:
        try:
            min_dist = min_dist_dict[(ele_2, ele_1)]
        except KeyError, e:
            return True
        else:
            return dist >= min_dist
    else:
        return dist >= min_dist


def remove_collision_within_region(reg, min_dist_dict):
    """Given a list of atoms, removes collision within the region.

    Args:
        reg (nparray): Atoms, array of record arrays.
        min_dist_dict (dict): A dictionary where the key is tuple of atom type
            names and value is the minimum distance in angstrom.

    Returns:
        nparray: The region of atoms with collisions removed: array of record
            arrays.
    """
    i = 0
    while (i < len(reg) - 1):
        prev = reg[0:i + 1]
        atm = reg[i]
        test = reg[i + 1:]
        is_safe = np.array(map(lambda x: apart_by_safe_distance(
            min_dist_dict, x, atm), test))
        test = test[np.array(is_safe)]
        reg = np.concatenate((prev, test))
        i += 1
    return reg


def remove_collision_between_regions(reg_1, reg_2, min_dist_dict):
    """Given two list of atoms, removes collision between two regions.

    Args:
        reg_1 (nparray): Atoms, array of record arrays.
        reg_2 (nparray): Atoms, array of record arrays.
        min_dist_dict (dict): A dictionary where the key is tuple of atom type
            names and value is the minimum distance in angstrom.

    Returns:
        nparray: reg_2 with all collision removed: array of record arrays.
    """
    for atm in reg_1:
        if len(reg_2) <= 0:
            break
        is_safe = np.array(map(lambda x: apart_by_safe_distance(
                               min_dist_dict, x, atm), reg_2))
        reg_2 = reg_2[is_safe]
    return reg_2


def remove_collision_surface_pair(struct, boundary_radius, min_dist_dict,
                                  dir_vec, random_delete=False):
    """Removes collisions on opposite surfaces of a lattice vector set.

    Args:
        struct (Structure obj): The Structure object to remove collision.
        boundary_radius (float): A proportion such that on each direction 
            atoms within the distance of this proportion of lattice vector
            length will be considered boundary atoms. 
        min_dist_dict (dict): A dictionary where the key is tuple of atom type
            names and value is the minimum distance in angstrom.
        dir_vec (nparray): A direction vector (3). Possible values are 
            [1., 0., 0.], [0., 1., 0.], and [0., 0., 1.].
        random_delete (bool, optional): When set to true, shuffle the list 
            of atoms before removing collision.

    Returns:
        int: Number of atoms removed.
    """
    orig_atom_count = len(struct.cartesian)

    on_btm_idx = np.logical_and(
        np.dot(struct.direct['position'], dir_vec) < (0.0 + boundary_radius),
        np.dot(struct.direct['position'], dir_vec) > (0.0 - boundary_radius))
    on_top_idx = np.logical_and(
        np.dot(struct.direct['position'], dir_vec) < (1.0 + boundary_radius),
        np.dot(struct.direct['position'], dir_vec) > (1.0 - boundary_radius))
    btm_atoms = struct.cartesian[on_btm_idx]
    if random_delete:
        np.random.shuffle(btm_atoms)
    top_atoms = struct.cartesian[on_top_idx]
    if random_delete:
        np.random.shuffle(top_atoms)
    struct.cartesian = struct.cartesian[np.logical_not(
        np.logical_or(on_btm_idx, on_top_idx))]
    coord = np.dot(dir_vec, struct.coordinates)
    top_atoms['position'] -= coord

    # Remove atoms that are too close to each other within bottom or top slice.
    top_atoms = remove_collision_within_region(top_atoms, min_dist_dict)
    btm_atoms = remove_collision_within_region(btm_atoms, min_dist_dict)

    # Remove atom collisions
    btm_atoms = remove_collision_between_regions(top_atoms, btm_atoms,
                                                 min_dist_dict)
    top_atoms['position'] += coord

    struct.cartesian = np.concatenate((struct.cartesian, btm_atoms))
    struct.cartesian = np.concatenate((struct.cartesian, top_atoms))
    struct.reconcile(according_to='C')

    final_atom_count = len(struct.cartesian)
    print(('%d atoms removed on surface on direction of ' +
           str(dir_vec) + ' .') % (orig_atom_count - final_atom_count))
    return orig_atom_count - final_atom_count


def remove_collision_on_interface(struct, boundary_radius, min_dist_dict,
                                  random_delete=False):
    """Removes collisions on grain boundary (the interface).

    Args:
        struct (Structure obj): The Structure object to remove collision.
        boundary_radius (float): A proportion such that on each direction 
            atoms within the distance of this proportion of lattice vector
            length will be considered boundary atoms. 
        min_dist_dict (dict): A dictionary where the key is tuple of atom type
            names and value is the minimum distance in angstrom.
        random_delete (bool, optional): When set to true, shuffle the list 
            of atoms before removing collision.

    Returns:
        int: Number of atoms removed.
    """
    orig_atom_count = len(struct.cartesian)

    on_iface_idx = np.logical_and(
        struct.direct['position'][:, 2] < (0.5 + boundary_radius),
        struct.direct['position'][:, 2] > (0.5 - boundary_radius))
    iface_atoms = struct.cartesian[on_iface_idx]
    if random_delete:
        np.random.shuffle(iface_atoms)
    struct.cartesian = struct.cartesian[np.logical_not(on_iface_idx)]
    struct.direct = struct.direct[np.logical_not(on_iface_idx)]
    iface_atoms = remove_collision_within_region(iface_atoms, min_dist_dict)

    struct.cartesian = np.concatenate((struct.cartesian, iface_atoms))
    struct.reconcile(according_to='C')

    final_atom_count = len(struct.cartesian)
    print('%d atoms removed on interface.' %
          (orig_atom_count - final_atom_count))
    return orig_atom_count - final_atom_count


def min_image_remove_collision(struct, min_dist_dict, random_delete=False):
    """Use minimum image convention algorithm to remove collision.

    Args:
        struct (Structure obj): The Structure object to remove collision.
        min_dist_dict (dict): A dictionary where the key is tuple of atom type
            names and value is the minimum distance in angstrom.
        random_delete (bool, optional): When set to true, shuffle the list 
            of atoms before removing collision.

    Returns:
        int: Number of atoms removed.
    """
    orig_atom_count = len(struct.direct)
    good_direct = []
    if random_delete:
        np.random.shuffle(struct.cartesian)
    for i in range(len(struct.cartesian)):
        qualified = True
        for j in range(len(good_direct)):
            if not qualified:
                break
            diff = good_direct[j]['position'] - struct.direct[i]['position']
            diff = np.array(map(lambda x: x + 1.0 if x < -0.5 else
                                (x - 1.0 if x > 0.5 else x), diff.tolist()))
            diff = np.linalg.norm(np.dot(diff, struct.coordinates))
            if not is_safe_distance(min_dist_dict, diff,
                                    good_direct[j]['element'], struct.direct[i]['element']):
                qualified = False

        if qualified:
            good_direct.append(struct.direct[i])

    good_direct = np.array(good_direct)
    struct.direct = good_direct
    struct.reconcile(according_to='D')
    final_atom_count = len(struct.direct)
    return final_atom_count - orig_atom_count


def remove_collision(struct, boundary_radius, min_dist_dict, fast=True,
                     random_delete=False):
    """Removes collision within a Structure object.

    Args:
        struct (Structure obj): The Structure object to remove collision.
        boundary_radius (float): A proportion such that on each direction 
            atoms within the distance of this proportion of lattice vector
            length will be considered boundary atoms. 
        min_dist_dict (dict): A dictionary where the key is tuple of atom type
            names and value is the minimum distance in angstrom.
        fast (bool, optional): When set to True, only consider boundaries; 
            otherwise, use the minimum image convention method.
        random_delete (bool, optional): When set to true, shuffle the list 
            of atoms before removing collision.

    Returns:
        int: Number of atoms removed.
    """
    orig_atom_count = struct.cartesian.shape[0]

    if fast:
        remove_collision_on_interface(struct, boundary_radius, min_dist_dict,
                                      random_delete)
        for dir_vec in np.identity(3):
            remove_collision_surface_pair(struct, boundary_radius,
                                          min_dist_dict, dir_vec,
                                          random_delete)
    else:
        min_image_remove_collision(struct, min_dist_dict)

    final_atom_count = struct.cartesian.shape[0]
    print('%d atoms removed in total.' % (orig_atom_count - final_atom_count))
    return orig_atom_count - final_atom_count
