from joblib import Parallel, delayed
from tqdm import tqdm
import numpy as np
import random
import colorednoise as cn

import util


# Specifications about the sources.
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

    def __init__(self, fwd, settings=DEFAULT_SETTINGS, parallel=False, n_jobs=-1):
        self.settings = settings
        self.check_settings()

        self.fwd = fwd
        self.parallel = parallel
        self.n_jobs = n_jobs

    def simulate(self, n_samples=10000):
        ''' Simulate sources and EEG data'''
        self.source_data = self.simulate_sources(n_samples)
        self.eeg_data = self.simulate_eeg()

    def simulate_sources(self, n_samples):
        print('Simulate Sources.')
        if self.parallel:
            source_data = np.stack(Parallel(n_jobs=self.n_jobs, backend='loky')
                (delayed(self.simulate_source)() for _ in range(n_samples)))
        else:
            source_data = np.stack([self.simulate_source() for _ in tqdm(range(n_samples))], axis=0)

        if self.settings['duration_of_trial'] > 0 :
            source_data = np.transpose(source_data, (1,0,2))
        else :
            source_data = source_data.T
        return source_data

    def simulate_source(self):
        ''' Returns a vector containing the dipole currents. Requires only a 
        dipole position list and the simulation settings.

        Parameters (located in the settings dict).
        ----------
        forward : From the forward object get the list of dipole positions
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
        source : numpy.ndarray, (n_dipoles x n_samples x timepoints) or (n_dipoles x n_samples) if duration_of_tril is 0
        , the simulated value of its dipole
        '''

        # Get a random sources number in range:
        number_of_sources = self.get_from_range(self.settings['number_of_sources'], dtype=int)

        # Get the diameter for each source
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

        # Get source centers
        src_centers = np.random.choice(np.arange(self.fwd.leadfield.shape[1]), \
            number_of_sources, replace=False)
        

        if self.settings['duration_of_trial'] > 0:
            signal_length = int(self.settings['sample_frequency']*self.settings['duration_of_trial'])
            # pulselen = self.settings['sample_frequency']/10
            # pulse = self.get_pulse(pulselen)
            
            signals = []
            for _ in range(number_of_sources):
                # Generate Gaussian 
                signal = cn.powerlaw_psd_gaussian(self.get_from_range(self.settings['beta'], dtype=float), signal_length) 
                
                signal /= np.max(np.abs(signal))

                signals.append(signal)
            
            sample_frequency = self.settings['sample_frequency']
        else:  # else its a single instance
            sample_frequency = 0
            signal_length = 1
            signals = [np.array([1])]*number_of_sources
        
        source = np.zeros((self.fwd.leadfield.shape[1], signal_length))

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

        return np.squeeze(source)

    def simulate_eeg(self):
        ''' Create EEG of specified number of trials based on sources and some SNR.
        Parameters
        -----------
        fwd : Forward
            the Forward object located in forward.py
        target_snr : tuple/list/float, 
                    desired signal to noise ratio. Can be a list or tuple of two 
                    floats specifying a range.
        beta : float
            determines the frequency spectrum of the noise added to the signal: 
            power = 1/f^beta. 
            0 will yield white noise, 1 will yield pink noise (1/f spectrum)
        n_jobs : int
                Number of jobs to run in parallel. -1 will utilize all cores.

        Return
        -------
        epochs : list
                list of either mne.Epochs objects or list of raw EEG data 
                (see argument <return_raw_data> to change output)
        '''
        print('Simulate EEG.')
        n_simulation_trials = 20
         
        # Desired Dim of sources: (samples x dipoles x time points)
        sources = self.source_data

         # if there is no temporal dimension...
        if len(sources.shape) < 3:
            # ...add empty temporal dimension
            sources = np.expand_dims(sources, axis=2)
        
        n_samples, _, _ = sources.shape
        n_elec = self.fwd.leadfield.shape[0]
        eeg_clean = np.array(self.project_sources(sources))

        # for now, I have to add noise.
        return eeg_clean.squeeze()

    def project_sources(self, sources):
        ''' Project sources through the leadfield to obtain the EEG data.
        Parameters
        ----------
        sources : numpy.ndarray
            3D array of shape (n_dipoles x n_samples x timepoints)
        
        Return the eeg signlas
        ------

        '''
        print('Project sources to EEG.')
        leadfield = self.fwd.leadfield
        n_dipoles ,n_samples, n_timepoints = sources.shape
        n_elec = leadfield.shape[0]

        # Collapse last two dims into one
        short_shape = (sources.shape[0], sources.shape[1]*sources.shape[2])

        sources_tmp = sources.reshape(short_shape)

        # Scale to allow for lower precision
        scaler = 1/sources_tmp.max()
        sources_tmp *= scaler

        # Perform Matmul
        result = np.matmul(leadfield.astype(np.float32), sources_tmp.astype(np.float32))
        # Reshape result
        result = result.reshape(result.shape[0], n_samples, n_timepoints)

        # Rescale
        result /= scaler

        return result

    def check_settings(self):
        ''' Check if settings are complete and insert missing 
            entries if there are any.
        '''
        # Check for wrong keys:
        for key in self.settings.keys():
            if not key in DEFAULT_SETTINGS.keys():
                msg = f'key {key} is not part of allowed settings. See DEFAULT_SETTINGS for reference: {DEFAULT_SETTINGS}'
                raise AttributeError(msg)
        
        # Check for missing keys and replace them from the DEFAULT_SETTINGS
        for key in DEFAULT_SETTINGS.keys():
            # Check if setting exists and is not None
            if not (key in self.settings.keys() and self.settings[key] is not None):
                self.settings[key] = DEFAULT_SETTINGS[key]


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


    @staticmethod
    def get_pulse(pulse_len):
        ''' Returns a pulse of given length. A pulse is defined as 
        half a revolution of a sine.
        
        Parameters
        ----------
        x : int
            the number of data points

        '''
        pulse_len = int(pulse_len)
        freq = (1/pulse_len) / 2
        time = np.arange(pulse_len)

        signal = np.sin(2*np.pi*freq*time)
        return signal
    
    