import numpy as np
from pymatgen.core import Lattice, Structure
from ase.io import read

cif = '/Users/agjnyiri/VS Code/Northwestern Projects/NaAsSe2 Analysis/Structures/Substitution Analysis/gamma_doped.cif'

atoms = read(cif)

As_indices = [atom.index for atom in atoms if atom.symbol == 'As']
Se_indices = [atom.index for atom in atoms if atom.symbol == 'Se']

distance_matrix = [None, None, None, None, None, None, None, None]
distance_vector_matrix = [None, None, None, None, None, None, None, None]

for idx, ii in enumerate(As_indices):
    distance_matrix[idx] = atoms.get_distances(ii, Se_indices, mic=True)
    distance_vector_matrix[idx] = atoms.get_distances(ii, Se_indices, mic=True, vector=True)

mean_bond_length_list = []
mean_bond_angle_list = []

#pyramid_indices = [[14, 10, 1], [15, 11, 0], [9, 0, 13], [8, 1, 12], [3, 7, 10], [2, 6, 11], [4, 12, 7], [5, 13, 6]]

for idx, distance_array in enumerate(distance_matrix):
    coordinated_Se_distances = sorted(distance_array)[:3]

    coordinated_Se_indices = [i for i, v in sorted(enumerate(distance_array), key=lambda x: x[1])][:3]

    coordinated_Se_vectors = distance_vector_matrix[idx][coordinated_Se_indices]

    #coordinated_Se_distances = distance_matrix[idx][pyramid_indices[idx]]
    #coordinated_Se_vectors = distance_vector_matrix[idx][pyramid_indices[idx]]

    coordinated_Se_angles = []
    
    pairs = [(0, 1), (0, 2), (1, 2)]
    
    for ii, jj in pairs:
        v1, v2 = coordinated_Se_vectors[ii], coordinated_Se_vectors[jj]
        cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
        coordinated_Se_angles.append(angle)

    label_num = idx + 1
    mean_bond_length = np.mean(coordinated_Se_distances)
    mean_bond_angle = np.mean(coordinated_Se_angles)


    #print("As #" + str(label_num) + ":")
    #print("Mean Bond Length: " + str(mean_bond_length))
    #print("Mean Bond Angle: " + str(mean_bond_angle) + "\n")

    mean_bond_length_list.append(mean_bond_length)
    mean_bond_angle_list.append(mean_bond_angle)

overall_mean_bond_length = np.mean(mean_bond_length_list)
overall_mean_bond_angle = np.mean(mean_bond_angle_list)

bond_length_std_dev = np.std(mean_bond_length_list)

print("Overall: ")
print("Mean Bond Length: " + str(overall_mean_bond_length))
print("Mean Bond Angle: " + str(overall_mean_bond_angle))
print("Bond Std. Dev.: " + str(bond_length_std_dev) + "\n")
