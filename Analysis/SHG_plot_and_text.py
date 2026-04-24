#!/usr/bin/env python3

import sys
import numpy as np
import matplotlib.pyplot as plt
from ase.io import read

materials = ['70_SCALP_NaAsSe2', '80_SCALP_NaAsSe2', '90_SCALP_NaAsSe2', '100_SCALP_NaAsSe2']

for material in materials:

    ######### Read in input file and initialize data #########

    out_file = "/projects/p32764/SHG/NaAsSe2/SCALP/" + material + ".gpw"

    atoms = read(out_file)

    cell = atoms.get_cell()
    cellsize = atoms.cell.cellpar()

    ######### Setup polarization permutations #########

    #polarizations = ['xxx', 'xxy', 'xxz', 'xyy', 'xyz', 'xzz', 'yxx', 'yxy', 'yxz', 'yyy', 'yyz', 'yzz', 'zxx', 'zxy', 'zxz', 'zyy', 'zyz', 'zzz']

    polarizations = ['xxx']

    ######### Save a plot for each tensor component and a text file with all tensor components #########
                
    w_des_1 = 0 # eV
    w_des_2 = 0.6 # eV
    chi_at_w_des_1 = {}
    chi_at_w_des_2 = {}
    s = ""

    for pol in polarizations:
        plt.figure(figsize=(6.0, 4.0), dpi=300)

        name = '/projects/p32764/SHG/NaAsSe2/SCALP/' + material + '_SHG_lg_' + pol + '.npy'
        shg = np.load(name)
        w_l = shg[0].real

        plt.plot(w_l, np.real(shg[1] * 1e12), '-')
        plt.plot(w_l, np.imag(shg[1] * 1e12), '--')

        legls = []
        legls.append('Real')
        legls.append('Imaginary')
        plt.xlabel(r'$\hbar\omega$ (eV)')
        plt.ylabel(r'$\chi_{' + pol + r'}\ \mathrm{(pm/V)}$')
        plt.legend(legls, ncol=2)
        
        plt.tight_layout()
        plt.savefig(material + '_SHG_lg_' + pol + '.png', dpi=300)
        plt.close()

        real_shg = np.real(shg[1] * 1e12) # pm/V
        imag_shg = np.imag(shg[1] * 1e12)

        ii = np.argmin(np.abs(w_l - w_des_1))
        chi_value_real_1 = np.real(shg[1][ii] * 1e12)  # pm/V

        jj = np.argmin(np.abs(w_l - w_des_2))
        chi_value_real_2 = np.real(shg[1][jj] * 1e12)  # pm/V
        
        chi_at_w_des_1[pol] = chi_value_real_1
        chi_at_w_des_2[pol] = chi_value_real_2

        if abs(chi_value_real_1) > 0.1:
            s = s + f"chi2_{pol} (real) = {round(chi_value_real_1, 1)} pm/V at hbar*w {round(w_l[ii], 1)} eV\n"
            s = s + f"chi2_{pol} (real) = {round(chi_value_real_2, 1)} pm/V at hbar*w {round(w_l[jj], 1)} eV\n"

    with open(material + '_SHG_lg_' + pol + '.txt', "w") as f:
        f.write(s)
