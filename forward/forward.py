import sys
import os  

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import numpy as np
import duneuropy as dp

folder_input = os.path.join(parent,'duneuropy/Data')
folder_output = os.path.join(parent,'duneuropy/DataOut')

grid_filename = os.path.join(folder_input, 'realistic_tet_mesh_6c.msh')
tensor_filename = os.path.join(folder_input, 'realistic_6c.cond')
electrode_filename = os.path.join(folder_input, 'realistic_electrodes_fitted.txt')

# Create MEEG driver
# We create the driver object which will read the mesh along with the conductivity tensors from the provided files
config = {
    'type' : 'fitted',
    'solver_type' : 'cg',
    'element_type' : 'tetrahedron',
    'volume_conductor' : {
        'grid.filename' : grid_filename,
        'tensors.filename' : tensor_filename
    },
    'solver' : {
        'verbose' : 1
    }
}
driver = dp.MEEGDriver3d(config)

# (optional) load the transfer matrix
filename = os.path.join(folder_output, 'transfer_realistic_tet_cg.npy')

# shape: (73, 885214)
tm_eeg = np.load(filename, allow_pickle=True)


source_model_config = {
    'type' : 'venant',
    'numberOfMoments' : 3,
    'referenceLength' : 20,
    'weightingExponent' : 1,
    'relaxationFactor' : 1e-6,
    'restricted' : False,
    'mixedMoments' : False,
    'restrict' : True,
    'initialization' : 'closest_vertex'
    
}

# Define test dipole
dipoles = [dp.Dipole3d([23.4541, 30, 100.716], [1, 0, 0])]

# Apply the transfer matrix
lf = driver.applyEEGTransfer(tm_eeg, dipoles, {
                    'source_model' : source_model_config,
                    'post_process' : True,
                    'subtract_mean' : True
                })
solution = np.array(lf[0])



