# Compute EEG leadfield using the standard (CG-) FEM approach,
# in a realistic volumetric tetrahedral 6 compartment head model
# with different source models (Partial integration, St. Venant, Whitney and Subtraction)
# using the transfer matrix apporach

# I. Import libraries
import numpy as np
import duneuropy as dp
import os

# II. Define input files
folder_input = '/path/to/input/'
folder_output = '/path/to/output/'
grid_filename = os.path.join(folder_input, 'realistic_tet_mesh_6c.msh')
tensor_filename = os.path.join(folder_input, 'realistic_6c.cond')
electrode_filename = os.path.join(folder_input, 'realistic_electrodes_fitted.txt')

# III. Create MEEG driver
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
# Compute the transfer matrix
transfer_config = {
    'solver' : {
        'reduction' : 1e-10
    }
}
tm = driver.computeEEGTransferMatrix(transfer_config)
tm_eeg = np.array(tm[0])

# (optional) save the transfer matrix
filename = os.path.join(folder_output, 'transfer_realistic_tet_cg.npy')
np.save(filename, tm_eeg)

# (optional) load the transfer matrix
# filename = os.path.join(folder_input, 'transfer_realistic_tet_cg.npy')
# tm_eeg = np.load(filename, allow_pickle=True)

# Create source model configurations (Partial integration St. Venant, Subtraction, Whitney)
source_model_configs = {
    'Partial integration' : {
        'type' : 'partial_integration'
    },
    'Venant' : {
        'type' : 'venant',
        'numberOfMoments' : 3,
        'referenceLength' : 20,
        'weightingExponent' : 1,
        'relaxationFactor' : 1e-6,
        'restricted' : False,
        'mixedMoments' : False,
        'restrict' : True,
        'initialization' : 'closest_vertex'
    },
    'Whitney' : {
        'type' : 'whitney',
        'referenceLength' : 20,
        'restricted' : True,
        'faceSources' : 'all',
        'edgeSources'  : 'all',
        'interpolation' : 'PBO'
    },
    'Subtraction' : {
        'type' : 'subtraction',
        'intorderadd' : 2,
        'intorderadd_lb' : 2
    }
}

# Define test dipole
dipoles = [dp.Dipole3d([23.4541, 30, 100.716], [1, 0, 0])]

# Apply the transfer matrix
solutions = dict()
for sm in source_model_configs:
    lf = driver.applyEEGTransfer(tm_eeg, dipoles, {
                    'source_model' : source_model_configs[sm],
                    'post_process' : True,
                    'subtract_mean' : True
                })
    solutions[sm] = np.array(lf[0])


# VI. Vizualization of output (mesh, the first dipole and the resulting potential of this dipole at the electrodes)
driver.write({
    'format' : 'vtk',
    'filename' : os.path.join(folder_output, 'realistic_cg_tet_transfer_headmodel')
})

pvtk = dp.PointVTKWriter3d(dipoles[0])
pvtk.write(os.path.join(folder_output,'realistic_cg_tet_transfer_testdipole'))

pvtk = dp.PointVTKWriter3d(electrodes, True)
pvtk.addScalarData('potential', solutions['Venant'][0]) 
pvtk.write(os.path.join(folder_output, 'realistic_cg_tet_transfer_lf_venant'))

# print a list of relevant publications
driver.print_citations()
