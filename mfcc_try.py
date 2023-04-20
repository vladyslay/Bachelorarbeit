# Initialization
import matplotlib.pyplot as plt
from scipy.io import wavfile
import scipy
import scipy.fft
import scipy.signal
import numpy as np

#import helper_functions # local file helper_functions.py
def freq2mel(f): return 2595*np.log10(1 + (f/700))
def mel2freq(m): return 700*(10**(m/2595) - 1)
def stft(data, fs):
    windowing_fn = np.sin(np.pi*np.arange(0.5,window_length,1)/window_length)**2
    spectrogram_matrix = np.zeros([window_length,window_count],dtype=complex)
    for window_ix in range(window_count):    
        data_window = np.multiply(windowing_fn,data[window_ix*window_step+np.arange(window_length)])
        spectrogram_matrix[:,window_ix] = np.fft.fft(data_window)
    
    fft_length = int((window_length+1)/2)
    
    return 20*np.log10(0.2+np.abs(spectrogram_matrix[range(fft_length),:]))

filename = './templates/apple01.wav'

# read from storage
fs, data = wavfile.read(filename)
data = data[:]

window_length_ms = 30
window_step_ms = 20
spectrum_length = 5000
window_length = int(window_length_ms*fs/1000)
window_step = int(window_step_ms*fs/1000)
windowing_function = np.sin(np.pi*np.arange(0.5,window_length,1)/window_length)**2
total_length = len(data)


# choose segment from random position in sample
starting_position = np.random.randint(total_length - window_length)

data_vector = data[starting_position:(starting_position+window_length),]
window = data_vector*windowing_function
time_vector = np.linspace(0,window_length_ms,window_length)

spectrum = scipy.fft.rfft(window,n=spectrum_length)
frequency_vector = np.linspace(0,fs/2000,len(spectrum))

# downsample to 16kHz (that is, Nyquist frequency is 8kHz, that is, everything about 8kHz can be removed)
idx = np.nonzero(frequency_vector <= 8)
frequency_vector = frequency_vector[idx]
spectrum = spectrum[idx]

logspectrum = 20*np.log10(np.abs(spectrum))
cepstrum = scipy.fft.rfft(logspectrum)

ctime = np.linspace(0,0.5*1000*spectrum_length/fs,len(cepstrum))



spectrogram = scipy.signal.stft(data,fs)
window_count = spectrogram.shape[0]

    

# choose segment from random position in sample
starting_position = np.random.randint(total_length - window_length)

data_vector = data[starting_position:(starting_position+window_length),]
window = data_vector*windowing_function
time_vector = np.linspace(0,window_length_ms,window_length)

spectrum = scipy.fft.rfft(window,n=spectrum_length)
frequency_vector = np.linspace(0,fs/2000,len(spectrum))

# downsample to 16kHz (that is, Nyquist frequency is 8kHz, that is, everything about 8kHz can be removed)
idx = np.nonzero(frequency_vector <= 8)
frequency_vector = frequency_vector[idx]
spectrum = spectrum[idx]

logspectrum = 20*np.log10(np.abs(spectrum))


# filterbank
frequency_step_Hz = 500
frequency_step = int(len(spectrum)*frequency_step_Hz/8000)
frequency_bins = int(len(spectrum)/frequency_step+.5)

slope = np.arange(.5,frequency_step+.5,1)/(frequency_step+1)
backslope = np.flipud(slope)
filterbank = np.zeros((len(spectrum),frequency_bins))
filterbank[0:frequency_step,0] = 1
filterbank[(-frequency_step):-1,-1] = 1
for k in range(frequency_bins-1):
    idx = int((k+0.25)*frequency_step) + np.arange(0,frequency_step)
    filterbank[idx,k+1] = slope
    filterbank[idx,k] = backslope


melbands = 20
maxmel = freq2mel(8000)
mel_idx = np.array(np.arange(.5,melbands,1)/melbands)*maxmel
freq_idx = mel2freq(mel_idx)

melfilterbank = np.zeros((len(spectrum),melbands))
freqvec = np.arange(0,len(spectrum),1)*8000/len(spectrum)
for k in range(melbands-2):    
    if k>0:
        upslope = (freqvec-freq_idx[k])/(freq_idx[k+1]-freq_idx[k])
    else:
        upslope = 1 + 0*freqvec
    if k<melbands-3:
        downslope = 1 - (freqvec-freq_idx[k+1])/(freq_idx[k+2]-freq_idx[k+1])
    else:
        downslope = 1 + 0*freqvec
    triangle = np.max([0*freqvec,np.min([upslope,downslope],axis=0)],axis=0)
    melfilterbank[:,k] = triangle
    
melreconstruct = np.matmul(np.diag(np.sum(melfilterbank**2+1e-12,axis=0)**-1),np.transpose(melfilterbank))



melbands = 20
maxmel = freq2mel(8000)
mel_idx = np.array(np.arange(.5,melbands,1)/melbands)*maxmel
freq_idx = mel2freq(mel_idx)

melfilterbank = np.zeros((spectrogram.shape[1],melbands))
freqvec = np.arange(0,spectrogram.shape[1],1)*8000/spectrogram.shape[1]
for k in range(melbands-2):    
    if k>0:
        upslope = (freqvec-freq_idx[k])/(freq_idx[k+1]-freq_idx[k])
    else:
        upslope = 1 + 0*freqvec
    if k<melbands-3:
        downslope = 1 - (freqvec-freq_idx[k+1])/(freq_idx[k+2]-freq_idx[k+1])
    else:
        downslope = 1 + 0*freqvec
    triangle = np.max([0*freqvec,np.min([upslope,downslope],axis=0)],axis=0)
    melfilterbank[:,k] = triangle
    
melreconstruct = np.matmul(np.diag(np.sum(melfilterbank**2+1e-12,axis=0)**-1),np.transpose(melfilterbank))


logmelspectrogram = 10*np.log10(np.matmul(np.abs(spectrogram)**2,melfilterbank)+1e-12)

mfcc = scipy.fft.dct(logmelspectrogram)

import matplotlib as mpl
default_figsize = mpl.rcParamsDefault['figure.figsize']
mpl.rcParams['figure.figsize'] = [val*2 for val in default_figsize]


plt.imshow(np.transpose(mfcc),aspect='auto',origin='lower')
plt.xlabel('Time (s)')
#plt.ylabel('Quefrency (ms)')
#plt.axis([0, len(data)/fs, 0, 20])
plt.title('MFCCs over time')
plt.show()