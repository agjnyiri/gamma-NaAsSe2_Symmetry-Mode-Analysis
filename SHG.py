#!/usr/bin/env python3

import sys
import numpy as np
from gpaw import GPAW, PW, FermiDirac
from gpaw.nlopt.matrixel import make_nlodata
from gpaw.nlopt.shg import get_shg
from gpaw.mpi import world

######### Read in input file #########

gpw_file = sys.argv[1]
material = sys.argv[2]

######### Momentum matrix elements #########

nlodata = make_nlodata(gpw_file)
nlodata.write('/projects/p32764/SHG/NaAsSe2/SCALP/' + material + '_mml.npz')

######### Calculate SHG tensor components #########

eta = 0.05  # Broadening in eV
dw = 0.050 # eV
w_ls = np.arange(0.01, 6 + 0.5*dw, dw)  # eV

#polarizations = ['xxx', 'xxy', 'xxz', 'xyy', 'xyz', 'xzz', 'yxx', 'yxy', 'yxz', 'yyy', 'yyz', 'yzz', 'zxx', 'zxy', 'zxz', 'zyy', 'zyz', 'zzz']

polarizations = ['xxx']

band_to_test = int(sys.argv[3]) # Set to band position at 20 eV from eigenvalues text file
band_n = list(range(band_to_test + 1))

eshift = 0.64 # eV
#eshift = float(sys.argv[4])

for pol in polarizations:

    shg_name = '/projects/p32764/SHG/NaAsSe2/SCALP/' + material + '_SHG_lg_' + pol + '.npy'

    get_shg(nlodata, freqs=w_ls, eta=eta, pol=pol,
            gauge='lg', out_name=shg_name, eshift=eshift, band_n=band_n)
