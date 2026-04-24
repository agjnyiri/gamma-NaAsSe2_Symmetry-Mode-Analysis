#!/usr/bin/env python3

import sys
import numpy as np
from ase import Atoms, io, build
from ase.cell import Cell
from gpaw import GPAW, PW, Mixer
from gpaw.eigensolvers import Davidson
from gpaw.mpi import world

######### Read in input data and initialize parameters #########

fname = sys.argv[1]
material = sys.argv[2]
ecut = float(sys.argv[3])
kpts = [int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])]
#kpts_dense = [2*int(sys.argv[4]), 2*int(sys.argv[5]), 2*int(sys.argv[6])]
nbands = int(sys.argv[7])

atoms = io.read(fname)
atoms.set_pbc([True, True, True])

######### Set SCF settings #########

xc = 'PBE'
gamma = True
h = 0.15
occupations = {'name': 'fermi-dirac', 'width': 0.05, 'fixmagmom': False}
symmetry = {'point_group': False, 'time_reversal': True,  'symmorphic': False}

conv = {
    'density': 1.0e-5,
    'eigenstates': 1.0e-7, # eV^2 / electron
    'bands': "CBM+20"}

mixer = Mixer(beta=0.3, weight=50.0)  # Fast for most non-spinpolarized systems
eigensolver = Davidson(niter=2)

######### Initialize calculator and run ground state calculation #########

settings = dict(
    xc=xc,
    convergence=conv,
    symmetry=symmetry,
    mixer=mixer,
    kpts={"size": kpts, "gamma": gamma},
    nbands=f'{nbands}%',
    occupations=occupations,
    h=h,
    eigensolver=eigensolver,
    parallel={'kpt': 64, 'band': 4}
)

calc = GPAW(mode=PW(ecut), **settings)

atoms.calc = calc

atoms.get_potential_energy()

######### Find index of eigenvalue 20 eV above CBM at gamma point for SHG calculation #########

eigen = calc.get_eigenvalues(kpt=0)

E_F = calc.get_fermi_level()

conduction_eigen = [e for e in eigen if e > E_F]

CBM = min(conduction_eigen)

target = CBM + 20.0

eigen_save = []
reached_twenty = False

for ii, value in enumerate(eigen):
    if value >= target and not reached_twenty:
        eigen_save.append(f"{value:.6f} (index: {ii})")
        reached_twenty = True

    else:
        eigen_save.append(f"{value:.6f}")

######### Save data #########

calc.write('/projects/p32764/SHG/NaAsSe2/SCALP/' + material + '.gpw', mode='all')
np.savetxt(material + '_eigenvalues.txt', eigen_save, fmt='%s')
