import mne

import numpy as np

def unpack_fwd(fwd):
    """ Helper function that extract the most important data structures from the 
    mne.Forward object

    Parameters
    ----------
    fwd : mne.Forward
        The forward model object

    Return
    ------
    fwd_fixed : mne.Forward
        Forward model for fixed dipole orientations
    leadfield : numpy.ndarray
        The leadfield (gain matrix)
    pos : numpy.ndarray
        The positions of dipoles in the source model
    tris : numpy.ndarray
        The triangles that describe the source mmodel
    neighbors : numpy.ndarray
        the neighbors of each dipole in the source model

    """
    if fwd['surf_ori']:
        fwd_fixed = fwd
    else:
        fwd_fixed = mne.convert_forward_solution(fwd, surf_ori=True, force_fixed=True,
                                                    use_cps=True, verbose=0)
    tris = fwd['src'][0]['use_tris']
    leadfield = fwd_fixed['sol']['data']

    source = fwd['src']
    try:
        subject_his_id = source[0]['subject_his_id']
        pos_left = mne.vertex_to_mni(source[0]['vertno'], 0, subject_his_id, verbose=0)
        pos_right = mne.vertex_to_mni(source[1]['vertno'],  1, subject_his_id, verbose=0)
    except:
        subject_his_id = 'fsaverage'
        pos_left = mne.vertex_to_mni(source[0]['vertno'], 0, subject_his_id, verbose=0)
        pos_right = mne.vertex_to_mni(source[1]['vertno'],  1, subject_his_id, verbose=0)

    pos = np.concatenate([pos_left, pos_right], axis=0)

    return fwd_fixed, leadfield, pos, tris#
