import colorednoise as cn

import numpy as np
import random

import util

DEFAULT_SETTINGS = {
    'number_of_sources': (1, 20),
    'extents': (1, 50),
    'amplitudes': (1, 10),
    'shapes': 'both',
    'duration_of_trial': 0.2,
    'sample_frequency': 100,
    'target_snr': 4,
    'beta': (0, 3),
}
class Simulation:
    ''' Simulate EEG and sources data.
    '''

    def __init__(self, fwd):
        self.settings = DEFAULT_SETTINGS
        self.fwd = fwd

    def simulate_source(self):
        ''' Returns a vector containing the dipole currents. Requires only a 
        dipole position list and the simulation settings.

        Parameters
        ----------
        pos : numpy.ndarray
            (n_dipoles x 3), list of dipole positions.
        number_of_sources : int/tuple/list
            number of sources. Can be a single number or a list of two 
            numbers specifying a range.
        extents : int/float/tuple/list
            diameter of sources (in mm). Can be a single number or a list of 
            two numbers specifying a range.
        amplitudes : int/float/tuple/list
            the current of the source in nAm
        shapes : str
            How the amplitudes evolve over space. Can be 'gaussian' or 'flat' 
            (i.e. uniform) or 'both'.
        duration_of_trial : int/float
            specifies the duration of a trial.
        sample_frequency : int
            specifies the sample frequency of the data.
        
        Return
        ------
        source : numpy.ndarray, (n_dipoles x n_timepoints), the simulated 
            source signal
        simSettings : dict, specifications about the source.

        Grova, C., Daunizeau, J., Lina, J. M., Bénar, C. G., Benali, H., & 
            Gotman, J. (2006). Evaluation of EEG localization methods using 
            realistic simulations of interictal spikes. Neuroimage, 29(3), 
            734-753.
        '''

        # Get a random sources number in range:
        number_of_sources = self.get_from_range(self.settings['number_of_sources'], dtype=int)

        # Get amplitudes for each source
        extents = [self.get_from_range(self.settings['extents'], dtype=float) for _ in range(number_of_sources)]

        # Decide shape of sources
        if self.settings['shapes'] == 'both':
            shapes = ['gaussian', 'flat']*number_of_sources
            np.random.shuffle(shapes)
            shapes = shapes[:number_of_sources]
            if type(shapes) == str:
                shapes = [shapes]

        elif self.settings['shapes'] == 'gaussian' or self.settings['shapes'] == 'flat':
            shapes = [self.settings['shapes']] * number_of_sources
            
       # Get amplitude gain for each source (amplitudes come in nAm)
        amplitudes = [self.get_from_range(self.settings['amplitudes'], dtype=float) * 1e-9 for _ in range(number_of_sources)]

        src_centers = np.random.choice(np.arange(self.fwd.leadfield.shape[0]), \
            number_of_sources, replace=False)

        if self.settings['duration_of_trial'] > 0:
            signal_length = int(self.settings['sample_frequency']*self.settings['duration_of_trial'])
            pulselen = self.settings['sample_frequency']/10
            # pulse = self.get_pulse(pulselen)
            
            signals = []
            for _ in range(number_of_sources):
                signal = cn.powerlaw_psd_gaussian(self.get_from_range(self.settings['beta'], dtype=float), signal_length) 
                
                signal /= np.max(np.abs(signal))

                signals.append(signal)
            
            sample_frequency = self.settings['sample_frequency']

        else : # else its a single instance
            sample_frequency = 0
            signal_length = 1
            signals = [np.array([1])]*number_of_sources
        
        source = np.zeros((self.fwd.leadfield.shape[0], signal_length))

        ##############################################
        # Loop through source centers (i.e. seeds of source positions)
        for i, (src_center, shape, amplitude, signal) in enumerate(zip(src_centers, shapes, amplitudes, signals)):
            dists = np.sqrt(np.sum((self.fwd.dipoles - self.fwd.dipoles[src_center, :])**2, axis=1))
            d = np.where(dists<extents[i]/2)[0]

            if shape == 'gaussian':
                sd = np.clip(np.max(dists[d]) / 2, a_min=0.1, a_max=np.inf)  # <- works better
                activity = np.expand_dims(util.gaussian(dists, 0, sd) * amplitude, axis=1) * signal
                source += activity
            elif shape == 'flat':
                activity = util.repeat_newcol(amplitude * signal, len(d)).T
                if len(activity.shape) == 1:
                    if len(d) == 1:
                        activity = np.expand_dims(activity, axis=0)    
                    else:
                        activity = np.expand_dims(activity, axis=1)
                source[d, :] += activity 
            else:
                msg = BaseException("shape must be of type >string< and be either >gaussian< or >flat<.")
                raise(msg)

        return source


    @staticmethod
    def get_from_range(val, dtype=int):
        ''' If list of two integers/floats is given this method outputs a value in between the two values.
        Otherwise, it returns the value.
        
        Parameters
        ----------
        val : list/tuple/int/float

        Return
        ------
        out : int/float

        '''
        if dtype==int:
            rng = random.randrange
        elif dtype==float:
            rng = random.uniform
        else:
            msg = f'dtype must be int or float, got {type(dtype)} instead'
            raise AttributeError(msg)

        if isinstance(val, (list, tuple, np.ndarray)):
            out = rng(*val)
        elif isinstance(val, (int, float)):
            out = val
        return out