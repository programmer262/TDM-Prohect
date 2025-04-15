import numpy as np
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from pydub import AudioSegment
from pydub.playback import play
from scipy.signal import butter, filtfilt
from scipy.fftpack import dct, idct, fft
import os
import matplotlib.pyplot as plt
import struct




class PsychoacousticModel:
    def __init__(self, fs):
        """
        Initialize psychoacoustic model
        :param fs: Sampling frequency
        """
        self.fs = fs
        self.bark_bands = self.__init_bark_bands()
        self.threshold_quiet = self.__init_threshold_quiet()
        
    def __init_bark_bands(self):
        """Initialize critical bands in Bark scale"""
        return [
            20, 100, 200, 300, 400, 510, 630, 770, 920, 1080, 1270, 1480, 1720, 2000,
            2320, 2700, 3150, 3700, 4400, 5300, 6400, 7700, 9500, 12000, 15500, 20000
        ]
    
    def __init_threshold_quiet(self):
        """Initialize threshold in quiet according to ISO 226"""
        freq = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 15000, 20000]
        thresh = [78, 55, 38, 28, 13, 6, 8, 11, 15, 24, 70]  # dB SPL
        return {f: 10**(t/10) for f, t in zip(freq, thresh)}
    
    def hz_to_bark(self, freq):
        """Convert Hz to Bark scale"""
        return 13 * np.arctan(0.00076 * freq) + 3.5 * np.arctan((freq / 7500)**2)
    
    def get_critical_bands(self, fft_size):
        """Get frequency bins for each critical band"""
        bins = []
        fft_freqs = np.fft.rfftfreq(fft_size, 1/self.fs)
        max_idx = len(fft_freqs) - 1
        
        for i in range(1, len(self.bark_bands)):
            low = self.bark_bands[i-1]
            high = min(self.bark_bands[i], self.fs/2)  # Cap at Nyquist
            
            band_bins = []
            for j, freq in enumerate(fft_freqs):
                if low <= freq < high and j <= max_idx:
                    band_bins.append(j)
            
            if band_bins:
                bins.append(band_bins)
            
        return bins
    
    def spreading_function(self, bark_z):
        """
        Calculate spreading function in Bark domain
        Using simplified spreading function from MP3 model
        """
        a = -23.5
        b = 15.5
        c = -8.5
        dz = np.arange(-3, 8, 0.1)
        spreading_db = np.zeros_like(dz)
        
        for i, z in enumerate(dz):
            if z < 0:
                spreading_db[i] = a * z
            elif z <= 1:
                spreading_db[i] = b * z
            else:
                spreading_db[i] = c * z
        
        spreading = 10**(spreading_db/10)
        return dz, spreading
    
    def calculate_masking_threshold(self, frame, frame_size):
        """
        Calculate masking threshold for audio frame
        :param frame: Audio frame (time domain)
        :param frame_size: Frame size
        :return: Masking threshold per frequency bin
        """
        if len(frame) != frame_size:
            frame = np.pad(frame, (0, frame_size - len(frame)), mode='constant')
        
        window = np.hanning(frame_size)
        windowed_frame = frame * window
        
        fft_result = np.fft.rfft(windowed_frame)
        power_spectrum = np.abs(fft_result)**2
        
        critical_bands = self.get_critical_bands(frame_size)
        
        band_powers = []
        for band_bins in critical_bands:
            if band_bins:
                valid_bins = [b for b in band_bins if b < len(power_spectrum)]
                band_power = np.sum(power_spectrum[valid_bins]) if valid_bins else 1e-10
                band_powers.append(band_power)
            else:
                band_powers.append(1e-10)
        
        band_spl = 10 * np.log10(band_powers)
        
        dz, spreading = self.spreading_function(0)
        
        masked_threshold = np.zeros(len(np.fft.rfftfreq(frame_size, 1/self.fs)))
        
        for i, band_bins in enumerate(critical_bands):
            if not band_bins or i >= len(band_spl):
                continue
                
            if band_bins:
                center_idx = band_bins[len(band_bins)//2]
                if center_idx >= len(np.fft.rfftfreq(frame_size, 1/self.fs)):
                    continue
                center_freq = np.fft.rfftfreq(frame_size, 1/self.fs)[center_idx]
                center_bark = self.hz_to_bark(center_freq)
                
                quiet_thresh = self.get_threshold_quiet(center_freq)
                masker_power = 10**(band_spl[i]/10)
                
                for j in range(len(masked_threshold)):
                    bin_freq = np.fft.rfftfreq(frame_size, 1/self.fs)[j]
                    bin_bark = self.hz_to_bark(bin_freq)
                    
                    bark_dist = bin_bark - center_bark
                    idx = np.argmin(np.abs(dz - bark_dist))
                    
                    if idx < len(spreading):
                        masking = masker_power * spreading[idx]
                        masked_threshold[j] = max(masked_threshold[j], masking)
        
        for i in range(len(masked_threshold)):
            freq = np.fft.rfftfreq(frame_size, 1/self.fs)[i]
            quiet = self.get_threshold_quiet(freq)
            masked_threshold[i] = max(masked_threshold[i], quiet)
        
        return masked_threshold
    
    def get_threshold_quiet(self, freq):
        """Get threshold in quiet for a given frequency (interpolating between defined points)"""
        keys = sorted(self.threshold_quiet.keys())
        
        if freq <= keys[0]:
            return self.threshold_quiet[keys[0]]
        if freq >= keys[-1]:
            return self.threshold_quiet[keys[-1]]
        
        low_key = keys[0]
        high_key = keys[-1]
        
        for k in keys:
            if k <= freq:
                low_key = k
            if k >= freq and high_key == keys[-1]:
                high_key = k
        
        low_val = self.threshold_quiet[low_key]
        high_val = self.threshold_quiet[high_key]
        
        log_freq_ratio = np.log10(freq/low_key) / np.log10(high_key/low_key)
        log_low = np.log10(low_val)
        log_high = np.log10(high_val)
        log_result = log_low + log_freq_ratio * (log_high - log_low)
        
        return 10**log_result
    
    def perceptual_bit_allocation(self, mdct_coeffs, frame_size):
        """
        Allocate bits to MDCT coefficients based on psychoacoustic masking
        :param mdct_coeffs: MDCT coefficients
        :param frame_size: Frame size
        :return: Perceptual weighting factors for quantization
        """
        if mdct_coeffs.ndim > 1:
            weights = np.ones_like(mdct_coeffs)
            for i in range(mdct_coeffs.shape[0]):
                frame_coeffs = mdct_coeffs[i]
                frame_weights = self._compute_frame_weights(frame_coeffs, frame_size)
                weights[i] = frame_weights
            return weights
        else:
            return self._compute_frame_weights(mdct_coeffs, frame_size)
    
    def _compute_frame_weights(self, frame_coeffs, frame_size):
        """Compute weights for a single frame"""
        approx_frame = idct(frame_coeffs, type=2, norm='ortho')
        masking_threshold = self.calculate_masking_threshold(approx_frame, frame_size)
        
        mdct_power = frame_coeffs**2
        mdct_to_fft = min(len(mdct_power), len(masking_threshold))
        
        weights = np.ones_like(frame_coeffs)
        
        for i in range(mdct_to_fft):
            if masking_threshold[i] > 0:
                snr = mdct_power[i] / masking_threshold[i]
                if snr < 1:
                    weights[i] = 0.1
                elif snr < 10:
                    weights[i] = 0.3
                else:
                    weights[i] = 1.0
        
        weights = np.convolve(weights, np.ones(3)/3, mode='same')
        return weights