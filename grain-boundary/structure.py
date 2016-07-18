import sys
import copy
import numpy as np
import utilities as util
import geometry as geom
from lattice import Lattice


class Structure(object):
    """docstring for Structure"""

    def __init__(self, comment, scaling, coordinate, elements_provided, atoms):
        self.comment = comment
        self.scaling = scaling
        self.coordinate = coordinate
        self.elements_provided = elements_provided
        self.atoms = atoms
        self.cartesian = False

    def __str__(self):
        res = ""
        res += "=== STRUCTURE: \n"
        res += "*** Comment: \n  " + self.comment + "\n"
        res += "*** Scaling: \n  %.5f\n" % (self.scaling)
        res += "*** Coordinates: \n"
        rows = [["", "x", "y", "z"]]
        rows.append(["  a"] + map(lambda x: "%.5f" % x,
                                  self.coordinate[0].tolist()))
        rows.append(["  b"] + map(lambda x: "%.5f" % x,
                                  self.coordinate[1].tolist()))
        rows.append(["  c"] + map(lambda x: "%.5f" % x,
                                  self.coordinate[2].tolist()))
        res += util.tabulate(rows) + "\n"
        res += "*** Element Names: \n  "
        res += ("P" if self.elements_provided else "Not p") + "rovided\n"
        res += "*** In Cartesian System: \n  "
        res += ("Yes" if self.cartesian else "No") + "\n"
        res += "*** Atoms: \n"
        rows = []
        rows.append(["  a", "b", "c", "element"])
        for ent in self.atoms:
            rows.append(["  %.5f" % ent[0][0], "%.5f" % ent[0][1],
                         "%.5f" % ent[0][2], "%s" % ent[1]])
        res += util.tabulate(rows)
        return res

    def cut_by_lattice(self, lattice):
        distance = np.dot(lattice.direction,
                          np.transpose(self.atoms["position"]))
        self.atoms = self.atoms[np.where(distance < lattice.distance)]
        return

    def to_cartesian(self):
        if self.cartesian:
            return
        new_representation = np.dot(np.transpose(self.coordinate),
                                    np.transpose(self.atoms["position"]))
        self.atoms["position"] = self.scaling * \
            np.transpose(new_representation)
        self.cartesian = True
        return

    def to_coordinate(self):
        if not self.cartesian:
            return
        orig_representation = np.dot(np.linalg.inv(np.transpose(coordinate)),
                                     np.transpose(self.atoms["position"]))
        self.atoms["position"] = np.transpose(orig_representation)
        self.cartesian = False
        return

    @staticmethod
    def from_vasp(path):
        with open(path, 'r') as in_file:
            comment = in_file.readline().split()[0]
            scaling = float(in_file.readline())
            coordinate = np.array([map(float, in_file.readline().split()),
                                   map(float, in_file.readline().split()),
                                   map(float, in_file.readline().split())])

            next_line = in_file.readline().split()
            try:
                element_count = map(int, next_line)
            except ValueError as e:
                element_list = next_line
                next_line = in_file.readline().split()
                element_count = map(int, next_line)
                elements_provided = True
            else:
                element_list = map(str, range(len(element_count)))
                elements_provided = False
            finally:
                if (len(element_list) != len(element_count)):
                    raise ValueError(
                        "Element list and count lengths mismatch.")

            elements = [[name for _ in range(count)] for (name, count) in
                        zip(element_list, element_count)]
            elements = np.array(elements).flatten()

            next_line = in_file.readline().split()
            if (next_line[0] == "Selective"):
                next_line = in_file.readline().split()

            if (next_line[0] != "Direct" and next_line[0] != "D"):
                raise ValueError("Only Mode \"Direct\" supported.")

            atoms = []
            for _ in range(0, sum(element_count)):
                atoms.append(map(float, in_file.readline().split()[0:3]))
            atoms = np.array(zip(atoms, elements),
                             dtype=[("position", ">f4", 3), ("element", "|S5")])

        return Structure(comment, scaling, coordinate, elements_provided, atoms)

    def to_vasp(self, name):
        with open(name, 'w') as out_file:
            out_file.write(self.comment + "\n")
            out_file.write(str(self.scaling) + "\n")
            for vector in self.coordinate:
                out_file.write(" ".join(map(str, vector.tolist())) + "\n")

            # Generate element list, assuming well ordered.
            element_list = []
            element_count = []
            prev_element = None
            prev_count = None
            for ele in self.atoms["element"]:
                if (ele != prev_element):
                    if prev_element is not None:
                        element_list.append(prev_element)
                        element_count.append(prev_count)
                    prev_element = ele
                    prev_count = 1
                else:
                    prev_count += 1
            element_list.append(prev_element)
            element_count.append(prev_count)
            if self.elements_provided:
                out_file.write(" ".join(element_list) + "\n")
                out_file.write(" ".join(map(str, element_count)) + "\n")
            else:
                out_file.write(" ".join(element_list) + "\n")
                out_file.write(" ".join(map(str, element_count)) + "\n")

            out_file.write("Direct\n")
            for pos in self.atoms["position"]:
                out_file.write("%.16f  %.16f  %.16f\n" %
                               (pos[0], pos[1], pos[2]))
        return

    def to_xyz(self, name):
        orig_format = self.cartesian
        self.to_cartesian()
        with open(name, 'w') as out_file:
            out_file.write(str(self.atoms.shape[0]) + "\n")
            out_file.write(self.comment + "\n")
            rows = []
            for ent in self.atoms:
                rows.append(["%s" % ent[1], "%.16f" % ent[0][0],
                             "%.16f" % ent[0][1], "%.16f" % ent[0][2]])
            out_file.write(util.tabulate(rows))
        if not orig_format:
            self.to_coordinate()
        return

    def rotate(self, from_vector, to_vector):
        # This is forced to be done in Cartesian mode.
        self.to_cartesian()
        rotation_matrix = geom.get_rotation_matrix(from_vector, to_vector)
        self.atoms["position"] = np.transpose(
            np.dot(rotation_matrix, np.transpose(self.atoms["position"])))
        # Also apply the rotation matrix to the original coordinate vectors
        # in case that they are useful in the future.
        self.coordinate = np.transpose(
            np.dot(rotation_matrix, np.transpose(self.coordinate)))
        return

    @staticmethod
    def combine(struct_1, struct_2):
        # First check that elements_provided values same.
        if struct_1.elements_provided != struct_2.elements_provided:
            raise ValueError(
                "Cannot combine two structures: ",
                "They must have same information (element names) ",
                "provided or not provided."
            )
        # Both structs should be in Cartesian mode.
        struct_1.to_cartesian()
        struct_2.to_cartesian()
        new_comment = ("Combined: " + struct_1.comment + " + " +
                       struct_2.comment)
        new_atoms = np.hstack((struct_1.atoms, struct_2.atoms))
        new_struct = Structure(new_comment, 1.0, None,
                               struct_1.elements_provided, new_atoms)
        new_struct.cartesian = True
        return new_struct

    @staticmethod
    def cut_and_combine(struct_1, lattice_1, struct_2, lattice_2, d):
        # If two structures are in fact the same, make a deep copy so that
        # operations on one will not affect the other.
        if struct_1 == struct_2:
            struct_2 = copy.deepcopy(struct_1)
        # First cut two structures by lattices.
        struct_1.cut_by_lattice(lattice_1)
        struct_2.cut_by_lattice(lattice_2)
        # Force into Cartesian.
        struct_1.to_cartesian()
        struct_2.to_cartesian()
        # Then rotate the two structures to face each other.
        struct_1.rotate(lattice_1.direction, [0, 0, 1])
        struct_2.rotate(lattice_2.direction, [0, 0, -1])
        # Calculate the centroids of two cutting surfaces of the two structures
        # and recenter them.
        # TODO: Discuss the method used to calculate the "centroids."
        epsilon = 1.0 * 10 ** -5
        cut_face_z_1 = np.amax(struct_1.atoms["position"][:, 2])
        cut_face_atoms_1 = struct_1.atoms[
            np.where(abs(struct_1.atoms["position"][:, 2] - cut_face_z_1)
                     <= epsilon)]
        cut_face_center_1 = np.mean(cut_face_atoms_1["position"], axis=0)
        struct_1.atoms["position"] -= cut_face_center_1

        cut_face_z_2 = np.amin(struct_2.atoms["position"][:, 2])
        cut_face_atoms_2 = struct_2.atoms[
            np.where(abs(struct_2.atoms["position"][:, 2] - cut_face_z_2)
                     <= epsilon)]
        cut_face_center_2 = np.mean(cut_face_atoms_2["position"], axis=0)
        struct_2.atoms["position"] -= cut_face_center_2

        # Raise struct_2 by d.
        struct_2.atoms["position"][:, 2] += abs(d)

        # Combine two structures and recenter.
        new_struct = Structure.combine(struct_1, struct_2)
        # TODO: recenter.
        return new_struct


def main(argv):
    struct = Structure.from_vasp(argv[1])
    lattice = Lattice([1, 1, 0], 1.4)
    d = 2.5
    new_struct = Structure.cut_and_combine(struct, lattice, struct, lattice, d)
    new_struct.to_xyz("cut_and_combine.xyz")

if __name__ == '__main__':
    main(sys.argv)
