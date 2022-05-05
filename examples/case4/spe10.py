import numpy as np
from scipy.stats import hmean
import porepy as pp

# ------------------------------------------------------------------------------#

class Spe10(object):

# ------------------------------------------------------------------------------#

    def __init__(self):
        self.full_shape = (60, 220, 85)
        self.full_physdims = (365.76, 670.56, 51.816)

        self.N = 0
        self.n = 0

        self.gb = None

        self.perm = None
        self.layers_id = None
        self.partition = None

# ------------------------------------------------------------------------------#

    def create_gb(self, num_layer=1, perm_folder="./spe10_perm/"):
        self._compute_size(num_layer)
        self._create_gb()

# ------------------------------------------------------------------------------#

    def _compute_size(self, num_layer):
        if num_layer == 1:
            self.shape = list(self.full_shape[:2])
            self.physdims = list(self.full_physdims[:2])
        else:
            self.shape = list(self.full_shape[:2]) + [num_layer]
            thickness = self.full_physdims[2] / self.full_shape[2] * num_layer
            self.physdims = list(self.full_physdims[:2]) + [thickness]

        self.N = np.prod(self.shape)
        self.n = np.prod(self.shape[:2])

# ------------------------------------------------------------------------------#

    def _create_gb(self,):
        g = pp.CartGrid(self.shape, self.physdims)
        g.compute_geometry()

        # it's only one grid but the solver is build on a gb
        self.gb = pp.meshing.grid_list_to_grid_bucket([g])

# ------------------------------------------------------------------------------#

    def read_perm(self, layers, perm_folder="./spe10_perm/"):
        self.layers = np.sort(np.atleast_1d(layers))

        shape = (self.n, self.layers.size)
        perm_xx, perm_yy, perm_zz = np.empty(shape), np.empty(shape), np.empty(shape)
        layers_id = np.empty(shape)

        for pos, layer in enumerate(self.layers):
            perm_file = perm_folder + str(layer) + ".tar.gz"
            #perm_file = perm_folder + "small_0.csv"
            perm_layer = np.loadtxt(perm_file, delimiter=",")
            perm_xx[:, pos] = perm_layer[:, 0]
            perm_yy[:, pos] = perm_layer[:, 1]
            perm_zz[:, pos] = perm_layer[:, 2]
            layers_id[:, pos] = layer

        shape = self.n*self.layers.size
        perm_xx = perm_xx.reshape(shape, order="F")
        perm_yy = perm_yy.reshape(shape, order="F")
        perm_zz = perm_zz.reshape(shape, order="F")
        self.perm = np.stack((perm_xx, perm_yy, perm_zz)).T

        self.layers_id = layers_id.reshape(shape, order="F")
        return self.perm

# ------------------------------------------------------------------------------#

    def save_perm(self):

        names = ["log10_perm_xx", "log10_perm_yy", "log10_perm_zz", "layer_id",
                 "perm_xx", "perm_yy", "perm_zz"]

        # for visualization export the perm and layer id
        for _, d in self.gb:

            d[pp.STATE][names[0]] = np.log10(self.perm[:, 0])
            d[pp.STATE][names[1]] = np.log10(self.perm[:, 1])
            d[pp.STATE][names[2]] = np.log10(self.perm[:, 2])

            d[pp.STATE][names[3]] = self.layers_id

            d[pp.STATE][names[4]] = self.perm[:, 0] * pp.DARCY
            d[pp.STATE][names[5]] = self.perm[:, 1] * pp.DARCY
            d[pp.STATE][names[6]] = self.perm[:, 2] * pp.DARCY

        return names

# ------------------------------------------------------------------------------#

    def perm_as_dict(self):
        return {"kxx": self.perm[:, 0], "kyy": self.perm[:, 1], "kzz": self.perm[:, 2]}

# ------------------------------------------------------------------------------#
