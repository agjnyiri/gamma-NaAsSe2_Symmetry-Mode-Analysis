import sys
import numpy as np
import matplotlib.pyplot as plt
from gpaw import GPAW
from gpaw.dos import DOSCalculator
from collections import Counter

# Input parameters
scf_file = '/projects/p32764/SHG/NaAsSe2/Full/0_NaAsSe2.gpw'
material = '0_NaAsSe2_no_Na'

emin, emax = -3, 3 # Energy window (eV)
npts = 2001 # DOS resolution
broadening = 0.05 # Gaussian broadening (eV)

# Load ground state and calculate DOS
calc = GPAW(scf_file)
atoms = calc.atoms
fermi = calc.get_fermi_level()

counts = Counter(atoms.get_chemical_symbols())

unique_elements = sorted(list(set(atoms.get_chemical_symbols())))

energies_rel = np.linspace(emin, emax, npts)
#energies_abs = energies_rel + fermi

doscalc = DOSCalculator.from_calculator(calc)
tdos = doscalc.raw_dos(energies_rel, width=broadening)
pdos_data = {}
l_chars = 'spdf'

for key in list(pdos_data.keys()):
    el = key.split()[0]
    pdos_data[key] /= counts[el]

for atom in atoms:
    symbol = atom.symbol
    index = atom.index
    if symbol != 'Na':
        for l in range(4):
            l_char = l_chars[l]
            key = f'{symbol} {l_char}'
            if key not in pdos_data:
                pdos_data[key] = np.zeros(npts)
            try:
                pdos_data[key] += doscalc.raw_pdos(energies_rel, a=index, l=l, width=broadening)
            except RuntimeError:
                pass

# Plot
fig = plt.figure(figsize=(5, 6)) # Adjusted for single panel
ax_dos = fig.add_subplot(111)
#ax_dos.plot(tdos, energies_rel, color='k', lw=1, alpha=0.6, label='Total DOS')
#ax_dos.fill_betweenx(energies_rel, tdos, color='k', alpha=0.1)

colors = ['gray', 'lightgray', 'orange', 'moccasin'] # 'blue', 'lightblue',

plot_idx = 0
for key in sorted(pdos_data.keys()):
    data = pdos_data[key]
    if np.any(data):
        ax_dos.plot(data, energies_rel, color=colors[plot_idx % len(colors)], lw=2, label=key)
        plot_idx += 1

ax_dos.axhline(0.0, color='k', linestyle='--', lw=1)
ax_dos.set_xlabel('DOS', fontsize=12)
ax_dos.set_ylabel(r'$E$ (eV)', fontsize=14)
ax_dos.set_ylim(emin, emax)
ax_dos.legend(loc='upper right', frameon=True)
plt.suptitle(f'{material}', fontsize=16, y=0.95)

# Save
plt.savefig(f"{material}_PDOS.png", dpi=300, bbox_inches='tight')
