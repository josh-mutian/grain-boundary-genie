import numpy as np
from utilities import *
import sys
from lattice import Lattice

class Structure(object):
    """docstring for Structure"""
    def __init__(self, comment, scaling, coordinate, elements_provided, atoms):
        self.comment = comment
        self.scaling = scaling
        self.coordinate = coordinate
        self.elements_provided = elements_provided
        self.atoms = atoms

    def __str__(self):
        res = ""
        res += "=== STRUCTURE: \n"
        res += "*** Comment: \n  " + self.comment + "\n"
        res += "*** Scaling: \n  %.5f\n" % (self.scaling) 
        res += "*** Coordinates: \n"
        rows = [["", "x", "y", "z"]]
        rows.append(["  a"] + map(lambda x : "%.5f" % x, 
            self.coordinate[0].tolist()))
        rows.append(["  b"] + map(lambda x : "%.5f" % x, 
            self.coordinate[1].tolist()))
        rows.append(["  c"] + map(lambda x : "%.5f" % x, 
            self.coordinate[2].tolist()))
        res += tabulate(rows) + "\n"
        res += "*** Element Names: \n  "
        res += ("P" if self.elements_provided else "Not p") + "rovided\n"
        res += "*** Atoms: \n"
        res += "  a        b        c        element\n"
        for ent in self.atoms:
            res += "  %.5f  %.5f  %.5f  %s\n" % (ent[0][0], ent[0][1], 
                ent[0][2], ent[1])
        return res[0:-1]

    def cut_by_lattice(self, lattice):
        distance = np.dot(lattice.direction, np.transpose(self.atoms["position"]))
        self.atoms = self.atoms[np.where(distance < lattice.distance)]
        return

    @staticmethod
    def from_vasp(path):
        in_file = open(path, 'r')
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
                raise ValueError("Element list and count lengths mismatch.")

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
            dtype = [("position", ">f4", 3), ("element", "|S5")])
        in_file.close()

        return Structure(comment, scaling, coordinate, element_provided, atoms)
    

    def output_as_vasp(self, name):
        out_file = open(name, 'w')
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
            out_file.write("%.16f  %.16f  %.16f\n" % (pos[0], pos[1], pos[2]))


def main(argv):
    struct = Structure.from_vasp(argv[1])
    lattice = Lattice([1, 2, 0], 1.5)
    struct.cut_by_lattice(lattice)
    print(lattice)
    struct.output_as_vasp("test_out.vasp")


if __name__ == '__main__':
    main(sys.argv)
