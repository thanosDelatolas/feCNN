import sys
import os

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import numpy as np
import duneuropy as dp

import scipy.io

# Define input files
folder_input = os.path.join(parent,'duneuropy/Data')
folder_output = os.path.join(parent,'duneuropy/DataOut')
# grid_filename = os.path.join(folder_input, 'realistic_tet_mesh_6c.msh')
tensor_filename = os.path.join(folder_input, 'realistic_6c.cond')

realistic_head_model_filename = os.path.join(folder_input, 'realistic_head_model.mat')
electrode_filename = os.path.join(folder_input, 'realistic_electrodes_fitted.txt')
dipoles_filename = os.path.join(folder_input, 'dipoles.mat')


realistic_head_model = scipy.io.loadmat(realistic_head_model_filename)

cond = np.array([0.00043, 0.0000042, 0.00001512, 0.00179, 0.00033, 0.00014])

# Create MEEG driver
# We create the driver object which will read the mesh along with the conductivity tensors from the provided files
config = {
    'type' : 'fitted',
    'solver_type' : 'cg',
    'element_type' : 'tetrahedron',
    'volume_conductor' : {
        'grid' : {
            'elements' : realistic_head_model['elements'],
            'nodes' : realistic_head_model['nodes']
        },
        'tensors' : {
            'labels' : realistic_head_model['labels'],
            'tensors' : cond
        }
    },
    'solver' : {
        'verbose' : 1
    }
}
driver = dp.MEEGDriver3d(config)




