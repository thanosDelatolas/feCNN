import sys
import os

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import numpy as np
import duneuropy as dp

import scipy.io

# Define the folders
folder_input = os.path.join(parent,'duneuropy/Data')
folder_output = os.path.join(parent,'duneuropy/DataOut')

# Define input files
realistic_head_model_filename = os.path.join(folder_input, 'realistic_head_model.mat')
electrode_filename = os.path.join(folder_input, 'realistic_electrodes_fitted.txt')
dipoles_filename = os.path.join(folder_input, 'dipoles.mat')
tensor_filename = os.path.join(folder_input, 'py-tensors.mat')
ele_filename = os.path.join(folder_input, 'elements.mat')

# load the head model data
realistic_head_model = scipy.io.loadmat(realistic_head_model_filename)
labels = realistic_head_model['labels'] - 1
nodes =  realistic_head_model['nodes']
elements =  scipy.io.loadmat(ele_filename)['ele']
tensors = scipy.io.loadmat(tensor_filename)['tensors_py']

print('Elements:', '({0}, {1})'.format(len(elements),len(elements[0])))
print('Nodes:','({0}, {1})'.format(len(nodes),len(nodes[0])))
print('Labels:','({0}, {1})'.format(len(labels),len(labels[0])))
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



# driver.write({
#     'format' : 'vtk',
#     'filename' : 'head-model'
# })
