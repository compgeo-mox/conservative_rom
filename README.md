# A Reduced Basis Method for Darcy flow systems that ensures local mass conservation by using exact discrete complexes
### Wietse M. Boon and Alessio Fumagalli 

The [examples](./examples/) folder contains the source code for replicating the three test cases. See [arXiv pre-print](https://arxiv.org/abs/2205.15626).

# Abstract
 A solution technique is proposed for flows in porous media that guarantees local conservation of mass. We first compute a flux field to balance the mass source and then exploit exact co-chain complexes to generate a solenoidal correction. A reduced basis method based on proper orthogonal decomposition is employed to construct the correction and we show that mass balance is ensured regardless of the quality of the reduced basis approximation. The method is directly applicable to mixed finite and virtual element methods, among other structure-preserving discretization techniques, and we present the extension to Darcy flow in fractured porous media.

# Citing
If you use this work in your research, we ask you to cite the following publication [arXiv pre-print](https://arxiv.org/abs/2205.15626).

# PorePy and PyGeoN version
If you want to run the code you need to install [PorePy](https://github.com/pmgbergen/porepy) and [PyGeoN](https://github.com/compgeo-mox/pygeon) and might revert them.
Newer versions of may not be compatible with this repository.<br>
PorePy valid commit: b3d8441065ac2c56a5f548a34c57a49984865ad0 <br>
PyGeoN valid tag: v0.2.0

# License
See [license](./LICENSE).
