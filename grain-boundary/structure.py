import numpy as np
from utilities import *

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
            self.coordinate[0].tolist()[0]))
        rows.append(["  b"] + map(lambda x : "%.5f" % x, 
            self.coordinate[1].tolist()[0]))
        rows.append(["  c"] + map(lambda x : "%.5f" % x, 
            self.coordinate[2].tolist()[0]))
        res += tabulate(rows) + "\n"
        res += "*** Atoms: \n"
        res += "  a        b        c       element\n"
        for ent in self.atoms:
            res += "  %.5f  %.5f  %.5f  %s\n" % (ent[0][0], ent[0][1], 
                ent[0][2], ent[1])
        return res[0:-1]