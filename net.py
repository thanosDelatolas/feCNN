import mne
from copy import deepcopy

from . import util
class Net:
    ''' The neural network class that creates and trains the model.

        Attributes
        ----------
        fwd : mne.Forward
            the mne.Forward forward model class.
    '''

    def __init__(self, fwd):
        self.fwd = deepcopy(fwd)
        _, self.leadfield, _, _ = util.unpack_fwd(fwd)
        self.n_channels = self.leadfield.shape[0]
        self.n_dipoles =  self.leadfield.shape[1]