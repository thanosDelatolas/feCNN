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

realistic_head_model_filename = os.path.join(folder_input, 'realistic_head_model.mat')
electrode_filename = os.path.join(folder_input, 'realistic_electrodes_fitted.txt')
dipoles_filename = os.path.join(folder_input, 'dipoles.mat')
tensor_filename = os.path.join(folder_input, 'tensors.mat')


# head model properties
realistic_head_model = scipy.io.loadmat(realistic_head_model_filename)
labels = np.array(realistic_head_model['labels']) 
labels = labels - np.ones(labels.shape)
elements = np.array(realistic_head_model['elements'])[:-3]
nodes =  np.array(realistic_head_model['nodes'])

cond_ratio = 3.6;   #conductivity ratio according to Akhtari et al., 2002
cond_compacta = (10**-4)*np.array([8, 16, 24, 28, 31, 41, 55, 70, 83, 167, 330])
cc=4

conductivity = np.array([0.43, cond_compacta[cc], cond_ratio*cond_compacta[cc], 1.79, 0.33, 0.14])

tensors = scipy.io.loadmat(tensor_filename)['tensors']

tensors = np.ones((3,3))


# Create MEEG driver
# We create the driver object which will read the mesh along with the conductivity tensors from the provided files
config = {
    'type' : 'fitted',
    'solver_type' : 'cg',
    'element_type' : 'tetrahedron',
    'volume_conductor' : {
        'grid' : {
            'elements' : elements,
            'nodes' : nodes
        },
        'tensors' : {
            'labels' : labels ,
            'conductivities' : conductivity,
            #'tensors' : np.array(tensors)
        }
    },
    'solver' : {
        'verbose' : 1
    }
}
driver = dp.MEEGDriver3d(config)



driver.write({
    'format' : 'vtk',
    'filename' : 'test'
})
