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
tensor_filename = os.path.join(folder_input, 'py-tensors.mat')

ele_filename = os.path.join(folder_input, 'elements.mat')

# head model properties
realistic_head_model = scipy.io.loadmat(realistic_head_model_filename)


labels = realistic_head_model['labels'] - 1
nodes =  realistic_head_model['nodes']
elements =  scipy.io.loadmat(ele_filename)['ele']


cond_ratio = 3.6;   #conductivity ratio according to Akhtari et al., 2002
cond_compacta = (10**4)*np.array([8, 16, 24, 28, 31, 41, 55, 70, 83, 167, 330])
cc=4

conductivity = np.array([0.43, cond_compacta[cc], cond_ratio*cond_compacta[cc], 1.79, 0.33, 0.14])

tensors = scipy.io.loadmat(tensor_filename)['tensors_py']

print('Elements:', '({0}, {1})'.format(len(elements),len(elements[0])))
print('Nodes:','({0}, {1})'.format(len(nodes),len(nodes[0])))
print('Labels:','({0}, {1})'.format(len(labels),len(labels[0])))
print('Conductivities:','({0},)'.format(len(conductivity)))
print('Tensors:',tensors.shape)



# Create MEEG driver
# We create the driver object which will read the mesh along with the conductivity tensors from the provided files
config = {
    'type' : 'fitted',
    'solver_type' : 'cg',
    'element_type' : 'hexahedron',
    'volume_conductor' : {
        'grid' : {
            'elements' :  elements,
            'nodes' : nodes
        },
        'tensors' : {
            'labels' : labels ,
            'tensors' : tensors
        }
    },
    'solver' : {
        'verbose' : 1
    }
}
driver = dp.MEEGDriver3d(config)



driver.write({
    'format' : 'vtk',
    'filename' : 'head-model'
})
