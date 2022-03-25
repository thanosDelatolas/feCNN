import os
from joblib import Parallel, delayed
from tqdm import tqdm
import numpy as np
import random
import pandas as pd

import util


# Specifications about the sources.
DEFAULT_SETTINGS = {
    'number_of_sources': 1 , # (1,5) , int/tuple/list
    'extents': (21, 80), # int/float/tuple/list,   diameter of sources (in mm)
    'amplitudes': (5, 10), #  int/float/tuple/list, the electrical current of the source in nAm
    'shapes': 'gaussian', # str,  How the amplitudes evolve over space. Can be 'gaussian' for now.
}

class Simulation:
    ''' Simulate EEG and sources data.
    '''

    def __init__(self, fwd, settings=DEFAULT_SETTINGS, source_data=None, eeg_data=None, locations=None,
        snr_levels=np.arange(-10, 25, 5, dtype=int), target_snr=(False, 10),
        noisy_eeg=False,parallel=False, n_jobs=-1
        ):
        self.settings = settings
        self.check_settings()

        # randrange does not include upper limit.
        if isinstance(self.settings['number_of_sources'], (list, np.ndarray)):
            self.settings['number_of_sources'][1] += 1
        elif isinstance(self.settings['number_of_sources'], (tuple)):
            self.settings['number_of_sources'] = (self.settings['number_of_sources'][0], self.settings['number_of_sources'][1]+1)

        self.fwd = fwd
        self.parallel = parallel
        self.n_jobs = n_jobs

        # if source_data, eeg_data and locations are already known from previous simulations
        self.source_data = source_data 
        self.eeg_data = eeg_data 
        self.locations = locations

        self.noisy_eeg = noisy_eeg
        self.snr_levels =  snr_levels
        self.target_snr = target_snr

        if  self.target_snr[0] and  not self.noisy_eeg:
            raise AttributeError('If target snr is true, then noisy_eeg must be true too.')
        
        self.simulated = False
        # to keep track the centers of each simulation (for two sources simultaneously mostly)
        self.source_centers = []
    
    
    def create_dipoles_dataset(self, times_each_dipole):
        ''' This method creates a trainning dataset which has n_dipoles * times_each_dipole  source spaces. 
            Each dipole is selected times_each_dipole as a seed location while the electrical current of the dipole 
            and the extent of the activation are selected randomly.

            The shape of the simulated sources will be times_each_dipole * n_dipoles, n_dipoles.

            Hence, times_each_dipole * n_dipoles different source spaces will be created.
        '''
        if self.simulated:
            print('The data are already simulated.')
            return

        n_dipoles = self.fwd.leadfield.shape[1]
        n_samples = int(times_each_dipole * n_dipoles)
        print('Sources Simulation')
        sources = np.stack([self.simulate_source(src_center=dipole % n_dipoles) \
            for dipole in tqdm(range(n_samples))], axis=0)
        
        self.source_data = sources.T      
        self.eeg_data = self.simulate_eeg(self.noisy_eeg)
        
        self.simulated = True
    

    def create_region_dataset(self,first_dipole, last_dipole, n_samples=100000):
        ''' This method creates a simulation for a spesific region. 
        '''
        if self.simulated:
            print('The data are already simulated.')
            return

        assert last_dipole > first_dipole
        
        print('Creating simulation for dipoles',first_dipole,'to',last_dipole)
        sources = np.stack([self.simulate_source(src_center= (dipole % (last_dipole - first_dipole + 1)) + first_dipole) \
            for dipole in tqdm(range(n_samples))], axis=0)
        
        self.source_data = sources.T      
        self.eeg_data = self.simulate_eeg()
        
        self.simulated = True


    def create_large_dataset(self, times_each_dipole,dir_x,dir_y):
        ''' This method creates a trainning dataset which has n_dipoles * times_each_dipole  source spaces. 
            Each dipole is selected times_each_dipole as a seed location while the electrical current of the dipole 
            and the extent of the activation are selected randomly.

            The shape of the simulated sources will be times_each_dipole * n_dipoles, n_dipoles.

            Hence, times_each_dipole * n_dipoles different source spaces will be created.

            it stores each simulated source and eeg to a folder in order to user keras_preprocessing_custom for the
            trainning.
        '''
        n_dipoles = self.fwd.leadfield.shape[1]
        n_samples = int(times_each_dipole * n_dipoles)
        
        eeg = np.zeros((73,n_samples))

        print('Creating dataset with {} samles'.format(n_samples))

        for sample in tqdm(range(n_samples)):
            dipole = sample % n_dipoles

            # simulate source
            source = self.simulate_source(src_center=dipole % n_dipoles)
            # calculate eeg 
            eeg[:,sample] = np.array(self.project_sources(source, verbose=False))

            np.save(dir_y+'source_{}.npy'.format(sample+1), source * (1e14))
        
        self.eeg_data = eeg
        np.save(dir_x+'eeg.npy',eeg)

    
    def create_evaluate_dataset(self, n_samples=100, snr=5):
        ''' This method creates a dataset for evaluation.

            Each source center is selected randomly.

            At the final eeg signal gaussian white noise is added with the 
            given snr (in dB).
        '''
        n_dipoles = self.fwd.leadfield.shape[1]
            
        eeg = np.zeros((73,n_samples))
        sources = np.zeros((n_dipoles,n_samples))
        print('Creating evalation dataset with {} samles and with snr {} dB'.format(n_samples,snr))

        for sample in tqdm(range(n_samples)):

            # simulate source
            source = self.simulate_source()
            sources[:, sample] = source
            # calculate eeg 
            eeg_clean = np.array(self.project_sources(source, verbose=False))
            eeg[:,sample] = eeg_clean + self.add_noise_to_eeg(eeg_clean, snr)

        return eeg, sources


    def simulate(self, n_samples=10000):
        ''' Simulate sources and EEG data
            Simulate n_samples randomly.
        '''
        if self.simulated :
            print('The data are already simulated.')
            return
        
        self.source_centers.clear()

        self.source_data = self.simulate_sources(n_samples)
        self.eeg_data = self.simulate_eeg(self.noisy_eeg)        
        self.simulated = True


    def simulate_sources(self, n_samples):
        ''' Simulate n_samles sources randomly. 
        Each source center is selected randomly.
        '''
        if self.simulated :
            print('The data are already simulated.')

            return self.source_data
        print('Simulate Sources.')
        if self.parallel:
            source_data = np.stack(Parallel(n_jobs=self.n_jobs, backend='loky')
                (delayed(self.simulate_source)() for _ in range(n_samples)))
        else:
            source_data = np.stack([self.simulate_source() for _ in tqdm(range(n_samples))], axis=0)

        source_data = source_data.T
        return source_data


    def create_depth_evaluate_dataset(self, path_to_save_dataset):
        ''' Simulates randomly dipoles per each depth in the head
        '''

        if not os.path.isdir(path_to_save_dataset):
            os.mkdir(path_to_save_dataset)
            
        source_space = self.fwd.dipoles[:,:3]

        min_depth = np.min(source_space[:,-1])
        # depth is in z column
        df = pd.DataFrame(source_space, columns = ['x','y','z'])

        
        z_df = df.groupby(['z'])
        for z_val, grouped_values in tqdm(z_df):
            #dipoles = df.index[df['z'] == z_val].tolist()
            grouped_values.reset_index()
            source_data = []

            for dipole, _ in grouped_values.iterrows():
                source = self.simulate_source(src_center=dipole).reshape(self.fwd.dipoles.shape[0], 1)
                if len(source_data) == 0:
                    source_data = source
                else :
                    source_data = np.concatenate((source_data,source), axis=-1)
            
            eeg_data = self.simulate_eeg(sources=source_data, noisy_eeg=self.noisy_eeg, verbose=False)
            path = os.path.join(path_to_save_dataset, str(z_val-min_depth))

            if not os.path.isdir(path):
                os.mkdir(path)

            np.save(os.path.join(path,'eeg.npy'),eeg_data)
            np.save(os.path.join(path,'sources.npy'),source_data)



    def simulate_locations(self, n_samples):
        ''' Simulate sources and EEG data. This function does not store the 
        electrical current of the sources. It used to simulate data for the LocCNN (net.py).

        The noise added in the eeg must be specified in the constructor of the simulation object.
        '''
        if self.simulated :
            print('The data are already simulated.')
            return
        
        self.source_centers.clear()
        eeg = np.zeros((73,n_samples))
        # source locations in the 3d space
        locations = np.zeros((n_samples,3))
        for ii in tqdm(range(n_samples)):
            # appends source centers
           
            source = self.simulate_source().reshape(self.fwd.dipoles.shape[0], 1)
            locations[ii,:] = self.fwd.dipoles[ii,:3]
            eeg[:,ii] = np.squeeze(self.simulate_eeg(sources=source, noisy_eeg=self.noisy_eeg, verbose=False))

        self.eeg_data = eeg
        self.locations = locations


    def simulate_source(self, src_center=-1):
        ''' Returns a vector containing the dipole currents. Requires only a 
        dipole position list and the simulation settings.

        Parameters (located in the settings dict).
        ----------src_center
        forward : The forward object located in forward.py
        number_of_sources : int/tuple/list
            number of sources. Can be a single number or a list of two 
            numbers specifying a range.
        extents : int/float/tuple/list
            diameter of sources (in mm). Can be a single number or a list of 
            two numbers specifying a range.
        amplitudes : int/float/tuple/list
            the electrical current of the source in nAm
        shapes : str
            How the amplitudes evolve over space. Can be 'gaussian' for now.
        src_center : int, 
            if src_center is already decided. If not -1 then it will be chosen randomly.
        Return
        ------
        source : numpy.ndarray, (n_dipoles,), the simulated value of its dipole
        '''
        
        # Get a random sources number in range:
        number_of_sources = self.get_from_range(self.settings['number_of_sources'], dtype=int)

        # Get the diameter for each source
        extents = [self.get_from_range(self.settings['extents'], dtype=float) for _ in range(number_of_sources)]

        # Decide shape of sources
        if self.settings['shapes'] == 'gaussian':
            shapes = [self.settings['shapes']] * number_of_sources
        else :
            raise AttributeError('Only Gaussian shape is supported!')


        # Get amplitude gain for each source (amplitudes come in nAm)
        amplitudes = [self.get_from_range(self.settings['amplitudes'], dtype=float) * 1e-15 for _ in range(number_of_sources)]

        # Get source centers
        if src_center == -1 :
            src_centers = np.random.choice(np.arange(self.fwd.leadfield.shape[1]), \
                number_of_sources, replace=False)
        else :
            src_centers = [src_center]
            
        signal_length = 1
        signals = [np.array([1])]*number_of_sources
        
        source = np.zeros((self.fwd.leadfield.shape[1], signal_length))

        ##############################################
        # Loop through source centers (i.e. seeds of source positions)
        for i, (center, shape, amplitude, signal) in enumerate(zip(src_centers, shapes, amplitudes, signals)):
            dists = np.sqrt(np.sum((self.fwd.dipoles - self.fwd.dipoles[center, :])**2, axis=1))
            d = np.where(dists<extents[i]/2)[0]

            if shape == 'gaussian':
                sd = np.clip(np.max(dists[d]) / 2, a_min=0.1, a_max=np.inf)  # <- works better
                activity = np.expand_dims(util.gaussian(dists, 0, sd) * amplitude, axis=1) * signal
                source += activity
            else:
                msg = BaseException("shape must be of type >string< and be >gaussian<")
                raise(msg)
                
        # keep track the seed dipoles for each simulation
        self.source_centers.append(src_centers)
        return np.squeeze(source)


    def simulate_eeg(self, sources=None,noisy_eeg=False,verbose=True):
        ''' Create EEG of specified number of trials based on sources and some SNR.
        Parameters
        -----------
        fwd : Forward
            the Forward object located in forward.py (class object)
        
        sources : ndarray or None
            The sources that will generate the eeg. If None, the self.source_data will be used.

        noisy_eeg: boolean
            True if we want to preturb with AWGN the eeg data.
        
        target_snr: tuple(boolean, int)
            if we want to have a traget snr instead of a random snr.

        n_jobs : int
                Number of jobs to run in parallel. -1 will utilize all cores.

        Return
        -------
        simulated eeg data : numpy.ndarray
            3D array of shape (n_elec x n_samples x timepoints)
        '''
        if self.simulated :
            print('The data are already simulated.')
            return
        
        if verbose:
            print('Simulate EEG.')

        if sources is None:
            # Desired Dim of sources: (samples x dipoles x time points)
            sources = self.source_data

        # calculate eeg 
        eeg_clean = np.array(self.project_sources(sources, verbose))

        if noisy_eeg:
            eeg_noisy = np.zeros(eeg_clean.shape)

            if verbose:
                print('Add noise to eeg')
                rng = tqdm(range(sources.shape[1]))
            else:
                rng = range(sources.shape[1])

            for sample in rng:
                
                if self.target_snr[0]:
                    snr = self.target_snr[1]
                else :
                    # get random snr
                    snr = np.random.choice(self.snr_levels)

                # add noise to eeg
                eeg_noisy[:,sample] = self.add_noise_to_eeg(eeg_clean[:,sample], snr)

            return eeg_noisy
        
        else :
            return eeg_clean



    def project_sources(self, sources, verbose=True):
        ''' Project sources through the leadfield to obtain the EEG data.
        Parameters
        ----------
        sources : numpy.ndarray
            3D array of shape (n_dipoles x n_samples x timepoints)
        
        Return the eeg signlas
        ------
        '''
        if verbose:
            print('Project sources to EEG.')
        leadfield = self.fwd.leadfield

        if leadfield.shape[1] != sources.shape[0] :
            sources = sources.T

        # Scale to allow for lower precision
        # scaler = 1/sources_tmp.max()
        # sources_tmp *= scaler

        # Perform Matmul
        result = np.matmul(leadfield.astype(np.float32), sources.astype(np.float32))

        # Rescale
        # result /= scaler

        return result


    def add_noise_to_eeg(self,eeg, snr_db):
        ''' This function adds noise to the eeg signal
        '''
        #print('Add AWGN with snr {} dB'.format(snr_db))

        mean_power = np.mean(eeg ** 2)
        mean_power_db = 10*np.log10(mean_power)
        noise_db = mean_power_db - snr_db
        noise_watts = 10 ** (noise_db/10)

        # Generate noise with calculated power
        awgn = np.random.normal(0, np.sqrt(noise_watts), size=eeg.shape)

        return eeg + awgn


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
    
    