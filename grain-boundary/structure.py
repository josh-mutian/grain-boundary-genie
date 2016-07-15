import numpy as np
import utilities as util
import sys
import geometry as geom
from lattice import Lattice


class Structure(object):
    """docstring for Structure"""

    def __init__(self, comment, scaling, coordinate, elements_provided, atoms):
        self.comment = comment
        self.scaling = scaling
        self.coordinate = coordinate
        self.inverse = np.linalg.inv(np.transpose(coordinate))
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
        self.atoms["position"] = np.transpose(new_representation)
        self.cartesian = True
        return

    def to_coordinate(self):
        if not self.cartesian:
            return
        orig_representation = np.dot(self.inverse,
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
                element_provided = True
            else:
                element_list = map(str, range(len(element_count)))
                element_provided = False
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

        return Structure(comment, scaling, coordinate, element_provided, atoms)

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
        return

    @staticmethod
    def combine(struct1, struct2):
        raise NotImplementedError("Method combine() is not finished yet.")


def main(argv):
    struct = Structure.from_vasp(argv[1])
    struct.to_xyz("bef_rot.xyz")
    lattice = Lattice([1, 1, 0], 1.4)
    struct.cut_by_lattice(lattice)
    struct.rotate([1, 1, 0], [0, 0, 1])
    struct.to_xyz("aft_rot.xyz")

if __name__ == '__main__':
    main(sys.argv)
