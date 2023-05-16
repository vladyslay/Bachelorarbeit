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
from speech_recognition_ml import *



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
    correct_matches = 0
    incorrect_matches = 0
    execution_times = []
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
        execution_times.append(execution_time)
        print("Sample:", sample[0], " matched with template:", match[0], 'execution time:', execution_time)
        if sample[0] == match[0]:
            correct_matches += 1
        else:
            incorrect_matches += 1
    
    correctness = correct_matches/(correct_matches + incorrect_matches)
    print("Correctness:", correctness)
    #print('stop')
    return execution_times, correctness

# HMM recognition for prerecorded samples

def recognize_prerecorded_ml(mode, dataset='open_source'):
    
    # create templates, as for dtw
    templates = []
    for filename in [x for x in os.listdir(template_folder) if x.endswith('.wav')][:-1]:
        # loading files
        filepath = os.path.join(template_folder, filename)
        audio, sampling_freq = librosa.load(filepath)
        template_label = filename[:-6]

        if mode == 'MFCC':
            processed_template = process_signal(audio, sampling_freq, 'MFCC')
            templates.append((template_label, processed_template))
        elif mode == 'FFT':
            processed_template = process_signal(audio, sampling_freq, 'FFT')
            templates.append((template_label, processed_template))
        elif mode == 'FM':
            processed_template_mfcc = process_signal(audio, sampling_freq, 'MFCC')
            processed_template_fft = process_signal(audio, sampling_freq, 'FFT')
            templates.append((template_label, processed_template_mfcc, processed_template_fft))
    
    
    # create samples as for dtw
    samples = []
    for filename in [x for x in os.listdir(sample_folder) if x.endswith('.wav')][:-1]:
        # loading files
        filepath = os.path.join(sample_folder, filename)
        audio, sampling_freq = librosa.load(filepath)
        sample_label = filename[:-6]

        if mode == 'MFCC':
            processed_sample = process_signal(audio, sampling_freq, 'MFCC')
            samples.append((sample_label, processed_sample))
        elif mode == 'FFT':
            processed_sample = process_signal(audio, sampling_freq, 'FFT')
            samples.append((sample_label, processed_sample))
        elif mode == 'FM':
            processed_sample_mfcc = process_signal(audio, sampling_freq, 'MFCC')
            processed_sample_fft = process_signal(audio, sampling_freq, 'FFT')
            samples.append((sample_label, processed_sample_mfcc, processed_sample_fft))
    
    
    # create model
    hmm_models = []
    
    
    # train model
    if dataset == 'learn_phase':
        #! small dataset from learning phase
        train_start = time.time()
        hmm_trainer = HMMTrainer()
        hmm_trainer.train(templates)
        hmm_models.append((hmm_trainer, label))
        hmm_trainer = None
        train_end = time.time()
        train_time = train_start - train_end
    elif dataset == 'open_source':
        #! open source speech data
        #TODO add this open source speech data to project
        train_start = time.time()
        hmm_trainer = HMMTrainer()
        hmm_trainer.train(X)
        hmm_models.append((hmm_trainer, label))
        hmm_trainer = None
        train_end = time.time()
        train_time = train_start - train_end
    
    scores=[]
    recognition_times =[]
    for item in hmm_models:
      hmm_model, label = item
      recognition_time_start = time.time()
      score = hmm_model.get_score(samples[1])
      scores.append(score)
      index=np.array(scores).argmax()
      recognition_time_end = time.time()
      recognition_time = recognition_time_end - recognition_time_start
      recognition_times.append(recognition_time)
      
      # Print the output
      #TODO change the input file to template
      print("\nTrue:", label)
      print("Predicted:", hmm_models[index][1])
    
    return scores, train_time, recognition_times



#! functions for execution on embedded devise


def recognize(templates, mode='MFCC'):
    
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
