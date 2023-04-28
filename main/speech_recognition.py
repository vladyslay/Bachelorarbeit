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
# import argparse
# from hmmlearn import hmm
# import sys
# import time
# import pyaudio
# import math
# import audioop
from helperf import *
from record_process_audio import record_process_audio
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean


# Bereiche: [low,high] --> what are these ?
hf_range = [5000, 9000]
lf_range = [0, 10]


# Calc filter coefficients
hp_coeff = signal.bessel(2, 100, btype='highpass', output='sos', fs=RATE)
lp_coeff = signal.bessel(10, 11000, btype='lowpass', output='sos', fs=RATE)
win_han = signal.windows.hann(CHUNK)

# ***********************************************************

#! this code is based on a pre recorded audiofiles (to exclude errors of on/off-set setting)

# storage locations for audiofiles
template_folder = './templates'
sample_folder = './samples'


# testing functions
# processing templates
def process_signal(signal, sampling_freq):
    # filtering
    filtered = applyFilter(signal, win_han, lp_coeff, hp_coeff)

    # extracting mfcc features from templates
    mfcc_features_templates = mfcc(y=filtered, sr=sampling_freq)

    return mfcc_features_templates


# templates = {"label": np.ndarray}
templates = {}
for filename in [x for x in os.listdir(template_folder) if x.endswith('.wav')][:-1]:
    # loading files
    filepath = os.path.join(template_folder, filename)
    audio, sampling_freq = librosa.load(filepath)

    processed_template = process_signal(audio, sampling_freq)

    templates.update({filename[:-6]: process_signal})


def recognize(templates):
    sample = record_process_audio()
    sample = process_signal(sample, sampling_freq)

    recognized = False
    # dynamic time wrapping to pair corresponding frames
    # also return a distance between two time sequences - similarity measure
    # TODO find out a distance that means that no words are recognized
    for template in templates:
        distance, path = fastdtw(template, sample, dist=euclidean)

        match = template

    return recognized, match


'''
# little abstraction 
def recognize_commando(commandos):
    sample = record_process_audio()
    sample = process_signal(sample, sampling_freq) #TODO sampling freq
    recognized, match = recognize(commandos, sample)
    return recognized, match

def recognize_keyword():
    sample = record_process_audio()
    return
'''
'''
def recognize_feedback():
    sample = record_process_audio()
    return
'''


def training(keywords, commandos):
    # separate storage spaces for diff types of templates to reduce search spaces
    templates_keywords = {}
    templates_commandos = {}
    # record, process and store a template
    for keyword in keywords:
        # TODO make the func return, as soon as offset detected
        template = record_process_audio()
        template = process_signal(
            template, sampling_freq)  # TODO sampling freq
        templates_keywords.update({keyword: template})

    # record, process and store a commando
    for commando in commandos:
        template = record_process_audio()
        template = process_signal(template, sampling_freq)
        templates_commandos.update({commando: template})

    return templates_keywords, templates_commandos
