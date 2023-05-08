# module to develope and test speech recognition separately

import librosa
import os
import time
from helperf import *
from record_process_audio import record_process_audio
from scipy.spatial.distance import euclidean
from dtw import dtw as dtw_mfcc
from numpy.linalg import norm
from tslearn.metrics import dtw



# ***********************************************************

#! this code is based on a pre recorded audio files (to exclude errors of on/off-set setting)

# storage locations for audiofiles
template_folder = './templates'
sample_folder = './samples'







#! recognize for prerecorded audio samples

def recognize_prerecorded(mode, metric=(lambda x, y: norm(x - y, ord=1))):
    # extract, process and store templates
    # templates = [("label", np.ndarray)]
    templates = []
    for filename in [x for x in os.listdir(template_folder) if x.endswith('.wav')][:-1]:
        # loading files
        filepath = os.path.join(template_folder, filename)
        audio, sampling_freq = librosa.load(filepath)

        if mode == 'MFCC':
            processed_template = process_signal(audio, sampling_freq, 'MFCC')
            templates.append((filename[:-6], processed_template))
        elif mode == 'FFT':
            processed_template = process_signal(audio, sampling_freq, 'FFT')
            templates.append((filename[:-6], processed_template))
        elif mode == 'FM':
            processed_template_mfcc = process_signal(audio, sampling_freq, 'MFCC')
            processed_template_fft = process_signal(audio, sampling_freq, 'FFT')
            templates.append((filename[:-6], processed_template_mfcc, processed_template_fft))
    
    # plotting mfcc features
    '''
    mfcc_features = mfcc_features.T
    plt.matshow(mfcc_features)
    plt.title('MFCC')
    '''
    
    # extract, process and store samples
    # samples = [("label", np.ndarray)]
    samples = []
    for filename in [x for x in os.listdir(sample_folder) if x.endswith('.wav')][:-1]:
        # loading files
        filepath = os.path.join(sample_folder, filename)
        audio, sampling_freq = librosa.load(filepath)

        if mode == 'MFCC':
            processed_sample = process_signal(audio, sampling_freq, 'MFCC')
            samples.append((filename[:-6], processed_sample))
        elif mode == 'FFT':
            processed_sample = process_signal(audio, sampling_freq, 'FFT')
            samples.append((filename[:-6], processed_sample))
        elif mode == 'FM':
            processed_sample_mfcc = process_signal(audio, sampling_freq, 'MFCC')
            processed_sample_fft = process_signal(audio, sampling_freq, 'FFT')
            samples.append((filename[:-6], processed_sample_mfcc, processed_sample_fft))
        
    
    # recognize
    print('Start matching')
    for sample in samples:
        start = time.time()
        distances = []
        match = 'none'
        for template in templates:
            if mode == 'MFCC':
                distance, cost, acc_cost, path = dtw_mfcc(template[1].T, sample[1].T, dist=metric)
                distances.append((template[0], distance))
            elif mode == 'FM':
                distance_mfcc, cost, acc_cost, path = dtw_mfcc(template[1].T, sample[1].T, dist=metric)
                distance_fft = dtw(template[-1], sample[-1])
                distance_av = (distance_mfcc + distance_fft) / 2
                distances.append((template[0], distance_av))
            elif mode == 'FFT':
                distance_fft = dtw(template[-1], sample[-1])
                distances.append((template[0], distance_fft))
                
        match = min(distances, key= lambda t: t[1])
        end = time.time()
        execution_time = end - start
        print("Sample:", sample[0], " matched with template:", match[0], 'execution time:', execution_time)
    
    print('stop')
    return execution_time



#! functions for execution on embedded devise


def recognize(templates, mode):
    
    # record voice command and extract it's features
    sample = record_process_audio()
    sample = process_signal(sample, RATE, mode)

    
    # dynamic time wrapping to pair corresponding frames
    # also return a distance between two time sequences - similarity measure
    start = time.time()
    distances = []
    match = 'none'
    for template in templates:
        if mode == 'MFCC':
            distance, cost, acc_cost, path = dtw(template[1].T, sample[1].T, dist=lambda x, y: norm(x - y, ord=1))
            distances.append((template[0], distance))
        elif mode == 'FM':
            distance_mfcc, cost, acc_cost, path = dtw(template[1].T, sample[1].T, dist=lambda x, y: norm(x - y, ord=1))
            distance_fft = dtw(template[-1], sample[-1])
            distance_av = (distance_mfcc + distance_fft) / 2
            distances.append((template[0], distance_av))
        elif mode == 'FFT':
            distance_fft = dtw(template[-1], sample[-1])
            distances.append((template[0], distance_fft))
    match = min(distances, key= lambda t: t[1])
    end = time.time()
    execution_time = end - start            
    return match, execution_time


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


def training(keywords, commandos, mode):
    # separate storage spaces for diff types of templates to reduce search spaces
    templates_keywords = []
    templates_commandos = []
    # record, process and store a template
    for keyword in keywords:
        # TODO make the func return, as soon as offset detected
        template = record_process_audio()
        template = process_signal(
            template, RATE, mode) 
        templates_keywords.append((keyword, template))

    # record, process and store a commando
    for commando in commandos:
        template = record_process_audio()
        template = process_signal(template, RATE, mode)
        templates_commandos.append((commando, template))

    return templates_keywords, templates_commandos
