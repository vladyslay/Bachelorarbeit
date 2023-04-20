# module to develope and test speech recognition separately

from librosa.feature import mfcc
import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fftpack import fft
from scipy import signal
import os
import argparse
from hmmlearn import hmm
import sys
import time
import pyaudio
import math
import audioop 

#testing the fucking git

#**********************************************
# filtering and processing functions from main
def applyFilter(ein, win_han, lp, hp):
    
    #filter data window wise 
    hpf = signal.sosfilt(hp, ein) # high pass IIR filter
    lpf = signal.sosfilt(lp, hpf) # low pass IIR filter

    win = np.multiply(lpf, win_han) #in req.py win = lpf * win_han --> TODO: this is officially not from filtering but from applying fft 
    
    return(win)


def applyFFT(data_int, RATE, win_han, lp, hp): 

    #apply fft 
    fft_raw = fft(applyFilter(data_int, win_han, lp, hp), n=RATE)    #was wird hier eingesetzt 

    # fft postprocessing, abs spectrum and normalization 
    fft_abs = np.abs(fft_raw[0:11000])
    fft_norm = (fft_abs-np.min(fft_abs))*1/(np.max(fft_abs)-np.min(fft_abs))

    return fft_norm

def volumeFilter(data_raw):
	start_rms = audioop.rms(data_raw, 1) # This is a measure of the power in an audio signal
	# st_tres.append(start_rms)
	# mpst_tres = 3*(np.mean(st_tres[0:10]))
	# print(mpst_tres)
	
	# if start_rms >=40:
	print("volume filter, start rms", start_rms)

	return start_rms


# parameters from main
FRAME = 10      #ms
RATE = 44100    #Hz
#CHUNK = int(RATE * FRAME * pow(10, -3))
CHUNK = 4096
FORMAT = pyaudio.paInt16
CHANNELS = 1

#Bereiche: [low,high] --> what are these ? 
hf_range = [5000,9000]
lf_range = [0,10]


# Calc filter coefficients 
hp_coeff = signal.bessel(2, 100, btype='highpass', output='sos', fs=RATE)
lp_coeff = signal.bessel(10, 11000, btype='lowpass', output='sos', fs=RATE)
win_han = signal.windows.hann(CHUNK)

#***********************************************************

#! this code is based on a pre recorded audiofiles (to exclude errors of on/off-set setting)

#starage locations for audiofiles
template_folder = './templates'
sample_folder = './samples'

# processing templates
def process_templates():
    #
    return

templates = {}
for filename in [x for x in os.listdir(template_folder) if x.endswith('.wav')][:-1]:
    # loading files
    audio, sampling_freq = librosa.load(filename)
    
    #extracting mfcc features from templates
    mfcc_features_templates = mfcc(y=audio, sr=sampling_freq)
    
    print()

