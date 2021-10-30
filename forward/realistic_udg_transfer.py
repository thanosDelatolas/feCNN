# Compute EEG leadfield using the Unfitted Discontinuous Galerkin (UDG-) FEM approach,
# in a realistic 6 compartment head model
# with the partial integration source model
# using the transfer matrix apporach
# for a test dipole

# I. Import libraries
import sys
import os

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import duneuropy as dp
import numpy as np


# II. Define input files
folder_input = '../duneuropy/Data/'
folder_output = '../duneuropy/DataOut/'
filenames = {'skin_surf' : os.path.join(folder_input, 'skin.npy'),
             'white_surf' : os.path.join(folder_input, 'white.npy'),
             'gray_surf' : os.path.join(folder_input, 'gray.npy'),
             'spongiosa_surf' : os.path.join(folder_input, 'skull_spongiosa.npy'),
             'compacta_surf' : os.path.join(folder_input, 'skull_compacta.npy'),
             'CSF_surf' : os.path.join(folder_input, 'CSF.npy'),}
electrode_filename = os.path.join(folder_input,'realistic_electrodes_unfitted.txt')

# III. Create MEEG driver
# We create the driver object using the input levels sets, a hexahedral coarse mesh and the conductivities
imgs = {fn:np.load(filenames[fn]) for fn in filenames}
refinements = 1
elements = tuple([(x-1)/(2**refinements) for x in imgs['skin_surf'].shape])
config = {
    'type' : 'unfitted',
    'solver_type' : 'udg',
    'compartments' : 6,
    'volume_conductor' : {
        'grid' : {
            'cells' : elements,
            'upper_right' : (256,256,256),
            'lower_left' : (0,0,0),
            'refinements' : refinements
        }
    },
    'domain' : {
        'domains' : ['skin', 'white', 'gray', 'spongiosa', 'compacta', 'csf'],
        'skin.positions' : ['ieeeee'],
        'white.positions' : ['-i----'],
        'gray.positions' : ['--i---'],
        'spongiosa.positions' : ['---i--'],
        'compacta.positions' : ['----i-'],
        'csf.positions' : ['i----i'],
        'level_sets' : ['skin_surf', 'white_surf', 'gray_surf', 'spongiosa_surf', 'compacta_surf', 'CSF_surf']
    },
    'solver' : {
        'edge_norm_type' : 'houston',
        'penalty' : 4,
        'ghost_penalty' : 0.1,
        'scheme' : 'sipg',
        'weights' : 'tensorOnly',
        'conductivities' : [0.00043,0.00014,0.00033,0.00001512,0.0000042,0.00179],
        'intorderadd' : 3,
        'verbose' : 3,
        'smoother' : 'default'
    }
}
for k in config['domain']['level_sets']:
    config['domain'][k] = {
        'type' : 'image',
        'data' : imgs[k].astype(float, order='F')
    }
driver = dp.MEEGDriver3d(config)


# IV. Read and set electrode positions
electrodes = np.genfromtxt(electrode_filename)
electrodes = [dp.FieldVector3D(t) for t in electrodes.tolist()]
driver.setElectrodes(electrodes, {})


# V. Compute EEG leadfield
# Compute the transfer matrix
transfer_config = {
    'solver' : {
        'reduction' : 1e-10,
        'compartment': 0
    }
}
tm = driver.computeEEGTransferMatrix(transfer_config)
tm_eeg = np.array(tm[0])

# (optional) save the transfer matrix
filename = os.path.join(folder_output, 'transfer_realistic_udg.npy')
np.save(filename, tm_eeg)

# (optional) load the transfer matrix
#filename = os.path.join(folder_output, 'transfer_realistic_udg.npy')
#tm_eeg = np.load(filename, allow_pickle=True)

# Create source model configurations (Partial integration)
source_model_config = {
    'type' : 'partial_integration',
    'compartment': '2'
}

# Define test dipole
dipoles = [dp.Dipole3d([210, 163, 90], [0, -1, 0])]

# Apply the transfer matrix
x = driver.makeDomainFunction()
solutions = list()
lf = driver.applyEEGTransfer(tm_eeg, dipoles, {
                'source_model' : source_model_config,
                'post_process' : True,
                'subtract_mean' : True
            })
solution = np.array(lf[0])

# VI. Vizualization of output
# The tissue surfaces, the first dipole and the resulting potential of this dipole at the electrodes are exported.
driver.write({
    'format' : 'vtk',
    'filename' : os.path.join(folder_output,'realistic_udg_transfer_headmodel'),
    'mode' : 'boundary'
})
pvtk = dp.PointVTKWriter3d(dipoles[0])
pvtk.write(os.path.join(folder_output, 'realistic_udg_transfer_testdipole'))

pvtk = dp.PointVTKWriter3d(electrodes, True)
pvtk.addScalarData('potential', solution[0])
pvtk.write(os.path.join(folder_output,'realistic_udg_transfer_lf_pi'))

# Print a list of relevant publications
driver.print_citations()
