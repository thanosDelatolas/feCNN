# Compute EEG leadfield using the Discontinuous Galerkin (DG-) FEM approach,
# in a realistic volumetric tetrahedral 6 compartment head model
# with the partial integration source model
# by solving the system directly for a test dipole


# I. Import libraries
import os
import sys

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import numpy as np
import duneuropy as dp


# II. Define input files
folder_input = os.path.join(parent,'duneuropy/Data')
folder_output = os.path.join(parent,'duneuropy/DataOut')
grid_filename = os.path.join(folder_input, 'realistic_tet_mesh_6c.msh')
tensor_filename = os.path.join(folder_input, 'realistic_6c.cond')
electrode_filename = os.path.join(folder_input, 'realistic_electrodes_fitted.txt')

# III. Create MEEG driver
# We create the driver object which will read the mesh along with the conductivity tensors from the provided files
config = {
    'type' : 'fitted',
    'solver_type' : 'dg',
    'element_type' : 'tetrahedron',
    'volume_conductor' : {
        'grid.filename' : grid_filename,
        'tensors.filename' : tensor_filename
    },
    'solver' : {
        'verbose' : 1,
        'dg_smoother_type' : 'ssor',
        'edge_norm_type' : 'houston',
        'intorderadd' : '0',
        'penalty' : '20',
        'reduction' : '1e-10',
        'scheme' : 'sipg',
        'weights' : 'true',
    }
}
driver = dp.MEEGDriver3d(config)


# IV. Read and set electrode positions
# When projecting the electrodes, we choose the closest nodes
electrodes = np.genfromtxt(electrode_filename,delimiter=None) 
electrodes = [dp.FieldVector3D(t) for t in electrodes.tolist()]
electrode_config = {
    'type' : 'closest_subentity_center',
    'codims' : [3]
}
driver.setElectrodes(electrodes, electrode_config)

# V. Compute EEG leadfield
#Create source model configurations (partial integration)
source_model_config = {
    'type' : 'partial_integration'
}

# Define test dipole
dipoles = [dp.Dipole3d([23.4541, 30, 100.716], [1, 0, 0])]

# Compute leadfield for test dipole
x = driver.makeDomainFunction()
driver.solveEEGForward(dipoles[0], x, {
            'solver.reduction' : 1e-10,
            'source_model' : source_model_config,
            'post_process' : True,
            'subtract_mean' : True
        })
lf = driver.evaluateAtElectrodes(x)
lf -= np.mean(lf)

# VI. the potential is written out in vtk format
driver.write(x,{
            'format' : 'vtk',
            'filename' : os.path.join(folder_output, 'realistic_dg_tet_direct_solution_pi'),
        })

# print a list of relevant publications
driver.print_citations()
