#!/usr/bin/env python3

import sys
import numpy as np
from ase import Atoms, io, build
from ase.cell import Cell
from gpaw import GPAW, PW, FermiDirac
from gpaw.mpi import world
#from amlt import safe_kgrid_from_cell_volume

ecuts = [400, 450, 500, 550, 600, 650, 700, 750, 800]
kpds = [500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000]

######### Originally part of the AMLT-Auto trainer package #########

def safe_kgrid_from_cell_volume(atoms, kpoint_density):
    # tries to keep equi-planar spacing in k-space to match a KPD
    import numpy as np
    kpd = kpoint_density
    lengths_angles = atoms.cell.cellpar()
    vol = atoms.get_volume()
    lengths = lengths_angles[0:3]
    ngrid = kpd/vol # BZ volume = 1/cell volume (without 2pi factors)
    plane_density_mean = (ngrid * lengths[0] * lengths[1] * lengths[2]) ** (1 / 3)

    nkpt_frac  = np.zeros(3)
    for i, l in enumerate(lengths):
        nkpt_frac[i] = max(plane_density_mean / l, 1)
        
    nkpt       = np.floor(nkpt_frac)
    actual_kpd = vol * nkpt[0]*nkpt[1]*nkpt[2]
    
    if True:
        plane_densities = lengths*nkpt
        #print(plane_densities)
        # we want to start with the largest plane spacing, not the one closest to another integer
        check_order = np.argsort(plane_densities)
    else:
        delta_ceil = np.ceil(nkpt_frac)-nkpt_frac # measure of which axes are closer to a whole number
        check_order = np.argsort(delta_ceil) # we do this so we keep the grid as even as possible only rounding up when they are close

    i = 0
    if actual_kpd < kpd:
        if np.isclose(nkpt_frac[check_order[0]], nkpt_frac[check_order[1]]) and np.isclose(nkpt_frac[check_order[1]], nkpt_frac[check_order[2]]):
            nkpt[check_order[0]] = nkpt[check_order[0]] +1
            nkpt[check_order[1]] = nkpt[check_order[1]] +1
            nkpt[check_order[2]] = nkpt[check_order[2]] +1
            actual_kpd = vol * nkpt[0]*nkpt[1]*nkpt[2]
            i = 3

        elif np.isclose(nkpt_frac[check_order[0]], nkpt_frac[check_order[1]]):
            nkpt[check_order[0]] = nkpt[check_order[0]] +1
            nkpt[check_order[1]] = nkpt[check_order[1]] +1
            actual_kpd = vol * nkpt[0]*nkpt[1]*nkpt[2]
            i = 2

        elif np.isclose(nkpt_frac[check_order[1]], nkpt_frac[check_order[2]]):
            nkpt[check_order[0]] = nkpt[check_order[0]] +1
            actual_kpd = vol * nkpt[0]*nkpt[1]*nkpt[2]
            if actual_kpd < kpd:
                nkpt[check_order[1]] = nkpt[check_order[1]] +1
                nkpt[check_order[2]] = nkpt[check_order[2]] +1
                actual_kpd = vol * nkpt[0]*nkpt[1]*nkpt[2]
            i = 3

    while actual_kpd < kpd and i<=2:
        nkpt[check_order[i]] = nkpt[check_order[i]] +1
        actual_kpd = vol * nkpt[0]*nkpt[1]*nkpt[2]
        i+=1

    kp_as_ints = [int(nkpt[i]) for i in range(3)]
    return kp_as_ints


######### SCF calculation for convergence #########

convergence = {
        'eigenstates': 1.0e-7}  # eV^2/electron

def scf(atoms, kpts, ecut):
    atoms.set_pbc([True, True, True])

    calc = GPAW(mode=PW(ecut),
        xc='PBE',
        kpts={'size': kpts, 'gamma': True},
        convergence=convergence)

    atoms.calc = calc

    # Run ground state calculation
    E_tot = atoms.get_potential_energy() # eV

    return E_tot/len(atoms) # eV/atom


######### Loop over given e_cuts #########

fname = sys.argv[1]
material = sys.argv[2]
ecut_converged = ecuts[0]
ecut_bool = False
energy_per_atom_ecut = []

if len(sys.argv) >= 2:
    try:
        atoms = io.read(fname)
        file_read_ok = True
    except:
        file_read_ok = False
        
    if file_read_ok:
        for ii, ecut in enumerate(ecuts):
            kpts = safe_kgrid_from_cell_volume(atoms, 10000) # kpt density of 10000
            energy_per_atom_ecut.append(scf(atoms, kpts, ecut)) # eV/atom

            if ii > 0 and np.abs(energy_per_atom_ecut[ii] - energy_per_atom_ecut[ii-1]) < 0.0005: # Convergence criterion of 0.5 meV/atom
                if world.rank == 0:
                    print(f"\nSuccessful e_cut convergence with {ecut}.\n")
                ecut_converged = ecut
                ecut_bool = True
                break

        if ecut_bool == False and world.rank == 0:
            print("\nUnsuccessful e_cut convergence.\n")

    else:
        print(f'"{fname}" failed to read!')
        sys.exit()

else:
    print('\nMissing atomic structure file!\n')
    sys.exit()


######### Loop over given k-grid densities #########

kpt_bool = False
energy_per_atom_kpt = []

for ii, kpd in enumerate(kpds):
    kpts = safe_kgrid_from_cell_volume(atoms, kpd)
    energy_per_atom_kpt.append(scf(atoms, kpts, ecut_converged)) # eV/atom

    if ii > 0 and np.abs(energy_per_atom_kpt[ii] - energy_per_atom_kpt[ii-1]) < 0.0005: # Convergence criterion of 0.5 meV/atom
        if world.rank == 0:
            print(f"\nSuccessful k-grid convergence with ({kpts[0]}, {kpts[1]}, {kpts[2]}).\n")
            kpt_bool = True
        break

if kpt_bool == False and world.rank == 0:
    print("\nUnsuccessful k-grid convergence.\n")


######### Save convergence data #########
    
if world.rank == 0:
    try:
        if ecut_bool:
            ecuts_trimmed = ecuts[:len(energy_per_atom_ecut)]

            data = np.column_stack((ecuts_trimmed, energy_per_atom_ecut))
            np.savetxt(material + "_ecut_convergence.dat", data,
                       header="E_cut (eV)  Energy per Atom (eV)")
            
            print("\nSaved e_cut convergence data.")

        if kpt_bool:
            kpds_trimmed = kpds[:len(energy_per_atom_kpt)]
            kpts_trimmed = [[int(d) for d in safe_kgrid_from_cell_volume(atoms, kpd)] for kpd in kpds_trimmed]

            data = np.column_stack((kpts_trimmed, energy_per_atom_kpt))
            np.savetxt(material + "_kpt_convergence.dat", data,
                       header="k-grid  Energy per Atom (eV)")
            
            print("\nSaved k-grid convergence data.")
            
    except Exception as e:
        print(f"\nError saving files: {e}\n")
