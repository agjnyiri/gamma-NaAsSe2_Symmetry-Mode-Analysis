#!/usr/bin/env python3

import sys
from ase import Atoms, io
from ase.optimize import BFGS, CellAwareBFGS
from ase.constraints import FixSymmetry
from ase.io import write
from ase.units import GPa
from gpaw import GPAW, PW, Mixer
from gpaw.eigensolvers import Davidson
from ase.filters import FrechetCellFilter

# Read in script inputs (e.g., ~/Tools/full_relaxation.py 248117.cif PBE_gamma_NaAsSe2 700 4 8 4)
fname = sys.argv[1]
material = sys.argv[2]
ecut = float(sys.argv[3])
kpts = [int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])]

atoms = io.read(fname)
atoms.set_pbc([True, True, True])

# Set relaxation inputs
fmax = 0.001  # Max force in eV/Ang
xc = 'PBE'
gamma = True
relax_cell = True
h=0.15
maxstep = 0.2 # Ang

conv = {#'energy': 1.0e-4, # eV / electron
    'density': 1.0e-5,
    'eigenstates': 1.0e-7, # eV^2 / electron
    'bands': "occupied",
    'forces': 0.1 * fmax}

symmetry = {'point_group': False, 'time_reversal': True,  'symmorphic': False} # Turn off everything but time_reversal symmetry when freely relaxing non-spin polarized materials
occupations = {'name': 'fermi-dirac', 'width': 0.05, 'fixmagmom': False}

mixer = Mixer(beta=0.3, weight=50.0)  # Fast for most non-spinpolarized systems
eigensolver = Davidson(niter=2)

# Set GPAW calculator
settings = dict(
    xc=xc,
    convergence=conv,
    symmetry=symmetry,
    mixer=mixer,
    kpts={"size": kpts, "gamma": gamma},
    occupations=occupations,
    h=h,
    eigensolver=eigensolver,
)

calc = GPAW(mode=PW(ecut), **settings)

atoms.calc = calc

#atoms.set_constraint(FixSymmetry(atoms))

# Set BFGS calculator
bfgs_kwrds = dict( # For all relaxations
    alpha=30, # eV/Ang^2, an estimate of how much atoms resist being displaced, 30 is good for softer materials
    logfile = '/projects/p32764/SHG/NaAsSe2/' + material +'_full_relaxation.log',
    trajectory = '/projects/p32764/SHG/NaAsSe2/' + material + '_full_relaxation.traj',
    maxstep=maxstep)

mask = [True, True, True, True, True, True] # Allowed relaxation of shape components: x y z, (shear components)
cell_bfgs_kwrds = {
    'bulk_modulus': 30 * GPa, # A better guess gives faster convergence, but it's better to overestimate a little than to underestimate
    'long_output':True}

if not relax_cell:
    dyn = BFGS(atoms, **bfgs_kwrds)
else:
    ucf = FrechetCellFilter(atoms, mask = mask)
    ucf.exp_cell_factor = 1.0  # May be needed with older version of ASE
    dyn = CellAwareBFGS(ucf, **bfgs_kwrds, **cell_bfgs_kwrds)

# Run relaxation
dyn.run(fmax=fmax)

# Save final results
atoms.calc.write('/projects/p32764/SHG/' + material + '.gpw', mode='all')
io.write('/projects/p32764/SHG/' + material + '_full_relaxation_final.traj', atoms)
write(material + '_full_relaxed.cif', atoms)
