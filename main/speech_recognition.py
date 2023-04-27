# module to develope and test speech recognition separately

from librosa.feature import mfcc
import librosa
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
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
from helperf import *
from record_process_audio import record_process_audio
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean


#why's the fucking git not commiting????

#Bereiche: [low,high] --> what are these ? 
hf_range = [5000,9000]
lf_range = [0,10]


# Calc filter coefficients 
hp_coeff = signal.bessel(2, 100, btype='highpass', output='sos', fs=RATE)
lp_coeff = signal.bessel(10, 11000, btype='lowpass', output='sos', fs=RATE)
win_han = signal.windows.hann(CHUNK)

#***********************************************************

#! this code is based on a pre recorded audiofiles (to exclude errors of on/off-set setting)

#storage locations for audiofiles
template_folder = './templates'
sample_folder = './samples'


# testing functions
# processing templates
def process_template(signal, sampling_freq):
    #filtering
    filtered = applyFilter(signal, win_han, lp_coeff, hp_coeff)
    
    #extracting mfcc features from templates
    mfcc_features_templates = mfcc(y=filtered, sr=sampling_freq)
    
    return mfcc_features_templates

# templates = {"label": np.ndarray}
templates = {}
for filename in [x for x in os.listdir(template_folder) if x.endswith('.wav')][:-1]:
    # loading files
    filepath = os.path.join(template_folder, filename)
    audio, sampling_freq = librosa.load(filepath)
    
    processed_template = process_template(audio, sampling_freq)
    
    templates.update({filename[:-6]: process_template})
    
    
    
def recognize(templates, sample):
    # dynamic time wrapping to pair corresponding frames 
    # also return a distance between two time sequences - similarity measure
    for template in templates:
        distance, path = fastdtw(template, sample, dist=euclidean)
    
    return    


# little abstraction 
def recognize_commando():
    sample = record_process_audio()
    return

def recognize_keyword():
    sample = record_process_audio()
    return

'''
def recognize_feedback():
    sample = record_process_audio()
    return
'''


def training(keywords, commandos):
    # separate storage spaces for diff types of templates to reduce search spaces
    templates_keywords = {}
    templates_commandos = {}
    for keyword in keywords:
        template = record_process_audio()
        templates_keywords.update({keyword: template})
        
    for commando in commandos:
        template = record_process_audio()
        templates_commandos.update({commando: template})
        
    return templates_keywords, templates_commandos