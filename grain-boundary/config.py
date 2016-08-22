from structure import Structure
from math import pi as PI
import utilities as util
import json

class GbSetting(object):
    """docstring for GbSetting"""
    def __init__(self, orient_1, orient_2, twist_agl):
        self.orient_1 = orient_1
        self.orient_2 = orient_2
        self.twist_agl = twist_agl


class Config(object):
    """docstring for Config"""
    def __init__(self, arg):
        # Basic GB settings, required from user input.
        self.struct_1 = struct_1
        self.struct_2 = struct_2
        self.gb_settings = gb_settings # List of GbSetting objects.

        # Coincident point search.
        self.coincident_pts_tolerance = 4.0
        self.coincident_pts_search_step = 25

        # Lattice vector generation.
        self.lattice_vec_agl_range = (0, PI)
        self.atom_counts_range = (5000, 10000)

        # Collision removal.
        self.min_atom_dist = {}
        self.boundary_radius = 2.0
        self.random_delete_atom = False

        # Output format.
        self.output_dir = ""
        self.output_name_prefix = ""
        self.overwrite_protect = False

    @staticmethod
    def from_gbconfig_file(path):
