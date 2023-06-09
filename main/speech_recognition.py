# module to develope and test speech recognition separately

import librosa
import os
import time
from helperf import *
from record_process_audio import record_process_audio
from dtw import dtw
from scipy.spatial.distance import euclidean
from speech_recognition_ml import *
from scipy.spatial.distance import cityblock, braycurtis
from librosa.feature import mfcc
from scipy.signal.windows import hann



# ***********************************************************

#! this code is based on a pre recorded audio files (to exclude errors of on/off-set setting)









#! recognize for prerecorded audio samples

def recognize_prerecorded(mode, metric=cityblock, 
                          coef1=0.5, coef2=0.5, coef3=0, reshaping_factor=14, lpc_order=6, window = hann, data = 'fruits'):
    
    if data == 'fruits':
        # storage locations for audiofiles
        template_folder = './templates'
        sample_folder = './samples'
        
        # extract, process and store templates
        # templates = [("label", np.ndarray)]
        templates = []
        for filename in [x for x in os.listdir(template_folder) if x.endswith('.wav')][:-1]:
            # loading files
            filepath = os.path.join(template_folder, filename)
            audio, sampling_freq = librosa.load(filepath)
            print(sampling_freq)

            if mode == 'MFCC':
                processed_template = process_signal(audio, sampling_freq, 'MFCC', window=window)
                templates.append((filename[:-6], processed_template))
            elif mode == 'FFT':
                processed_template = process_signal(audio, sampling_freq, 'FFT', window=window)
                #print(processed_template.shape)
                templates.append((filename[:-6], processed_template))
            elif mode == 'FM':
                processed_template_mfcc = process_signal(audio, sampling_freq, 'MFCC', window=window)
                processed_template_fft = process_signal(audio, sampling_freq, 'FFT', window=window)
                templates.append((filename[:-6], processed_template_mfcc, processed_template_fft))
            elif mode == 'LPC':
                processed_template = process_signal(audio, sampling_freq, 'LPC', lpc_order=lpc_order, window=window)
                templates.append((filename[:-6], processed_template))
            elif mode == 'LM':
                processed_template_mfcc = process_signal(audio, sampling_freq, 'MFCC', window=window)
                processed_template_lpc= process_signal(audio, sampling_freq, 'LPC',  lpc_order=lpc_order, window=window)
                templates.append((filename[:-6], processed_template_mfcc, processed_template_lpc))
            elif mode == 'LMF':
                processed_template_mfcc = process_signal(audio, sampling_freq, 'MFCC', window=window)
                processed_template_fft = process_signal(audio, sampling_freq, 'FFT', window=window)
                processed_template_lpc= process_signal(audio, sampling_freq, 'LPC', lpc_order=lpc_order, window=window)
                templates.append((filename[:-6], processed_template_mfcc, processed_template_lpc, processed_template_fft))
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
                processed_sample = process_signal(audio, sampling_freq, 'MFCC', window=window)
                samples.append((filename[:-6], processed_sample))
            elif mode == 'FFT':
                processed_sample = process_signal(audio, sampling_freq, 'FFT', window=window)
                samples.append((filename[:-6], processed_sample))
            elif mode == 'FM':
                processed_sample_mfcc = process_signal(audio, sampling_freq, 'MFCC', window=window)
                processed_sample_fft = process_signal(audio, sampling_freq, 'FFT', window=window)
                samples.append((filename[:-6], processed_sample_mfcc, processed_sample_fft))
                #print('MFCC shape:', samples[-1][1].shape, 'FFT shape:', samples[-1][-1].shape)
            elif mode == 'LPC':
                processed_sample = process_signal(audio, sampling_freq, 'LPC', lpc_order=lpc_order, window=window)
                samples.append((filename[:-6], processed_sample))
            elif mode == 'LM':
                processed_sample_mfcc = process_signal(audio, sampling_freq, 'MFCC', window=window)
                processed_sample_lpc= process_signal(audio, sampling_freq, 'LPC', lpc_order=lpc_order, window=window)
                samples.append((filename[:-6], processed_sample_mfcc, processed_sample_lpc))
            elif mode == 'LMF':
                processed_sample_mfcc = process_signal(audio, sampling_freq, 'MFCC', window=window)
                processed_sample_fft = process_signal(audio, sampling_freq, 'FFT', window=window)
                processed_sample_lpc= process_signal(audio, sampling_freq, 'LPC', lpc_order=lpc_order, window=window)
                samples.append((filename[:-6], processed_sample_mfcc, processed_sample_lpc, processed_sample_fft))
        
        # recognize
        print('Start matching')
        correct_matches = 0
        incorrect_matches = 0
        execution_times = []
        for sample in samples:
            #print(sample[1].shape)
            start = time.time()
            distances = []
            match = 'none'
            for template in templates:
                if mode == 'MFCC':
                    distance, cost, acc_cost, path = dtw(template[1].T, sample[1].T, dist=metric)
                    distances.append((template[0], distance))
                elif mode == 'FM':
                    distance_mfcc, cost, acc_cost, path = dtw(template[1].T, sample[1].T, dist=metric)
                    distance_fft, cost, acc_cost, path = dtw(np.reshape(template[-1], (reshaping_factor, -1)), 
                                                                  np.reshape(sample[-1], (reshaping_factor, -1)), 
                                                                  dist=metric)
                    distance_av = distance_mfcc * coef1 + (distance_fft * 100) * coef2
                    distances.append((template[0], distance_av))
                elif mode == 'FFT':
                    distance = dtw(np.reshape(template[-1], (reshaping_factor, -1)), 
                                                                  np.reshape(sample[-1], (reshaping_factor, -1)), 
                                                                  dist=metric)
                    #distance_fft = dtw(template[-1], sample[-1])
                    distances.append((template[0], distance))
                elif mode == 'LPC':
                    distance, cost, acc_cost, path = dtw(np.reshape(template[1], (7, -1)), 
                                                                  np.reshape(sample[1], (7, -1)), dist=braycurtis)
                    distances.append((template[0], distance))
                elif mode == 'LM':
                    distance_mfcc, cost, acc_cost, path = dtw(template[1].T, sample[1].T, dist=metric)
                    distance_lpc, cost, acc_cost, path = dtw(np.reshape(template[-1], (7, -1)), 
                                                                  np.reshape(sample[-1], (7, -1)), dist=braycurtis)
                    #print(distance_mfcc)
                    #print(distance_lpc)
                    distance_av = distance_mfcc * coef1 + distance_lpc * coef2 * 1000
                    distances.append((template[0], distance_av))
                elif mode == 'LMF':
                    distance_mfcc, cost, acc_cost, path = dtw(template[1].T, sample[1].T, dist=metric)
                    distance_lpc, cost, acc_cost, path = dtw(np.reshape(template[2], (7, -1)), 
                                                                  np.reshape(sample[2], (7, -1)), dist=braycurtis)
                    distance_fft, cost, acc_cost, path = dtw(np.reshape(template[-1], (reshaping_factor, -1)), 
                                                                  np.reshape(sample[-1], (reshaping_factor, -1)), 
                                                                  dist=metric)
                    distance_av = distance_mfcc * coef1 + (distance_fft * 10) * coef2 + (distance_lpc * 1000) * coef3
                    print('coef1:', coef1, 'coef2:', coef2, 'coef3:', coef3)
                    print('dist mfcc:', distance_mfcc, '* coef=', distance_mfcc * coef1)
                    print('dist fft:', distance_fft, '*coef=', (distance_fft * 10) * coef2)
                    print('dist lpc', distance_lpc, '*coef=', (distance_lpc * 1000) * coef3)
                    print('distance av:', distance_av)
                    distances.append((template[0], distance_av))

            match = min(distances, key= lambda t: t[1])
            end = time.time()
            execution_time = end - start
            execution_times.append(execution_time)
            print("Sample:", sample[0], " matched with template:", match[0], 'execution time:', execution_time)
            if sample[0] == match[0]:
                correct_matches += 1
            else:
                incorrect_matches += 1
            #print('correct matches:', correct_matches, 'incorrect matches:', incorrect_matches)
        
                
    elif data == 'numbers':
        speakers_samples = []
        speaker_dep_samples = './speaker dependant/samples'
        speaker_dep_templates = './speaker dependant/templates'

        print('start extracting samples')
        for directory in os.listdir(speaker_dep_samples):
            filepath = os.path.join(speaker_dep_samples, directory)
            samples = []
            for filename in [x for x in os.listdir(filepath) if x.endswith('.wav')][:-1]:
                filepath1 = os.path.join(filepath, filename)
                audio, sampling_freq = librosa.load(filepath1)
                #print(sampling_freq)

                if mode == 'MFCC':
                    processed_sample = process_signal(audio, sampling_freq, 'MFCC', window=window)
                    samples.append((filename[5:-4], processed_sample))
                elif mode == 'FFT':
                    processed_sample = process_signal(audio, sampling_freq, 'FFT', window=window)
                    samples.append((filename[5:-4], processed_sample))
                elif mode == 'FM':
                    processed_sample_mfcc = process_signal(audio, sampling_freq, 'MFCC', window=window)
                    processed_sample_fft = process_signal(audio, sampling_freq, 'FFT', window=window)
                    samples.append((filename[5:-4], processed_sample_mfcc, processed_sample_fft))
                    #print('MFCC shape:', samples[-1][1].shape, 'FFT shape:', samples[-1][-1].shape)
                elif mode == 'LPC':
                    processed_sample = process_signal(audio, sampling_freq, 'LPC', lpc_order=lpc_order, window=window)
                    samples.append((filename[5:-4], processed_sample))
                elif mode == 'LM':
                    processed_sample_mfcc = process_signal(audio, sampling_freq, 'MFCC', window=window)
                    processed_sample_lpc= process_signal(audio, sampling_freq, 'LPC', lpc_order=lpc_order, window=window)
                    samples.append((filename[5:-4], processed_sample_mfcc, processed_sample_lpc))
                elif mode == 'LMF':
                    processed_sample_mfcc = process_signal(audio, sampling_freq, 'MFCC', window=window)
                    processed_sample_fft = process_signal(audio, sampling_freq, 'FFT', window=window)
                    processed_sample_lpc= process_signal(audio, sampling_freq, 'LPC', lpc_order=lpc_order, window=window)
                    samples.append((filename[5:-4], processed_sample_mfcc, processed_sample_lpc, processed_sample_fft))
            speakers_samples.append(samples)
        print('samples extracted')
        
        print('start extracting templates')
        speakers_templates = []
        for directory in os.listdir(speaker_dep_samples):
            filepath = os.path.join(speaker_dep_samples, directory)
            templates = []
            for filename in [x for x in os.listdir(filepath) if x.endswith('.wav')][:-1]:
                filepath1 = os.path.join(filepath, filename)
                audio, sampling_freq = librosa.load(filepath1)
                
                if mode == 'MFCC':
                    processed_template = process_signal(audio, sampling_freq, 'MFCC', window=window)
                    templates.append((filename[5:-4], processed_template))
                elif mode == 'FFT':
                    processed_template = process_signal(audio, sampling_freq, 'FFT', window=window)
                    #print(processed_template.shape)
                    templates.append((filename[5:-4], processed_template))
                elif mode == 'FM':
                    processed_template_mfcc = process_signal(audio, sampling_freq, 'MFCC', window=window)
                    processed_template_fft = process_signal(audio, sampling_freq, 'FFT', window=window)
                    templates.append((filename[5:-4], processed_template_mfcc, processed_template_fft))
                elif mode == 'LPC':
                    processed_template = process_signal(audio, sampling_freq, 'LPC', lpc_order=lpc_order, window=window)
                    templates.append((filename[5:-4], processed_template))
                elif mode == 'LM':
                    processed_template_mfcc = process_signal(audio, sampling_freq, 'MFCC', window=window)
                    processed_template_lpc= process_signal(audio, sampling_freq, 'LPC',  lpc_order=lpc_order, window=window)
                    templates.append((filename[5:-4], processed_template_mfcc, processed_template_lpc))
                elif mode == 'LMF':
                    processed_template_mfcc = process_signal(audio, sampling_freq, 'MFCC', window=window)
                    processed_template_fft = process_signal(audio, sampling_freq, 'FFT', window=window)
                    processed_template_lpc= process_signal(audio, sampling_freq, 'LPC', lpc_order=lpc_order, window=window)
                    templates.append((filename[5:-4], processed_template_mfcc, processed_template_lpc, processed_template_fft))
            speakers_templates.append(templates)
        print('templates extracted')
        
        
        # recognize
        print('Start matching')
        correct_matches = 0
        incorrect_matches = 0
        execution_times = []
        for samples in speakers_samples:
            i = 0
            for sample in samples:
                #print(sample[1].shape)
                start = time.time()
                distances = []
                match = 'none'
                for template in speakers_templates[i]:
                    if mode == 'MFCC':
                        distance, cost, acc_cost, path = dtw(template[1].T, sample[1].T, dist=metric)
                        distances.append((template[0], distance))
                    elif mode == 'FM':
                        distance_mfcc, cost, acc_cost, path = dtw(template[1].T, sample[1].T, dist=metric)
                        distance_fft, cost, acc_cost, path = dtw(np.reshape(template[-1], (reshaping_factor, -1)), 
                                                                      np.reshape(sample[-1], (reshaping_factor, -1)), 
                                                                      dist=metric)
                        distance_av = distance_mfcc * coef1 + (distance_fft * 100) * coef2
                        distances.append((template[0], distance_av))
                    elif mode == 'FFT':
                        distance = dtw(np.reshape(template[-1], (reshaping_factor, -1)), 
                                                                      np.reshape(sample[-1], (reshaping_factor, -1)), 
                                                                      dist=metric)
                        #distance_fft = dtw(template[-1], sample[-1])
                        distances.append((template[0], distance))
                    elif mode == 'LPC':
                        distance, cost, acc_cost, path = dtw(np.reshape(template[1], (lpc_order +1, -1)), 
                                                                      np.reshape(sample[1], (lpc_order +1, -1)), dist=braycurtis)
                        distances.append((template[0], distance))
                    elif mode == 'LM':
                        distance_mfcc, cost, acc_cost, path = dtw(template[1].T, sample[1].T, dist=metric)
                        distance_lpc, cost, acc_cost, path = dtw(np.reshape(template[-1], (7, -1)), 
                                                                      np.reshape(sample[-1], (7, -1)), dist=braycurtis)
                        #print(distance_mfcc)
                        #print(distance_lpc)
                        distance_av = distance_mfcc * coef1 + distance_lpc * coef2 * 1000
                        distances.append((template[0], distance_av))
                    elif mode == 'LMF':
                        distance_mfcc, cost, acc_cost, path = dtw(template[1].T, sample[1].T, dist=metric)
                        distance_lpc, cost, acc_cost, path = dtw(np.reshape(template[2], (7, -1)), 
                                                                      np.reshape(sample[2], (7, -1)), dist=braycurtis)
                        distance_fft, cost, acc_cost, path = dtw(np.reshape(template[-1], (reshaping_factor, -1)), 
                                                                      np.reshape(sample[-1], (reshaping_factor, -1)), 
                                                                      dist=metric)
                        distance_av = distance_mfcc * coef1 + (distance_fft * 10) * coef2 + (distance_lpc * 1000) * coef3
                        print('coef1:', coef1, 'coef2:', coef2, 'coef3:', coef3)
                        print('dist mfcc:', distance_mfcc, '* coef=', distance_mfcc * coef1)
                        print('dist fft:', distance_fft, '*coef=', (distance_fft * 10) * coef2)
                        print('dist lpc', distance_lpc, '*coef=', (distance_lpc * 1000) * coef3)
                        print('distance av:', distance_av)
                        distances.append((template[0], distance_av))

                match = min(distances, key= lambda t: t[1])
                end = time.time()
                execution_time = end - start
                execution_times.append(execution_time)
                print("Sample:", sample[0], " matched with template:", match[0], 'execution time:', execution_time)
                if sample[0] == match[0]:
                    correct_matches += 1
                else:
                    incorrect_matches += 1
                #print('correct matches:', correct_matches, 'incorrect matches:', incorrect_matches)    
            i+=1
    correctness = correct_matches/(correct_matches + incorrect_matches)
    
    execution_time = sum(execution_times)/len(execution_times)
    print("Correctness:", correctness)
    return execution_time, correctness



# HMM recognition for prerecorded samples
    '''
def recognize_prerecorded_ml(mode, dataset='open_source'):
    
    

    # create templates, as for dtw
    # [(label, feature1, feature2)]
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
    template_folder = 'hmm-speech-recognition-0.1/audio'
    for dirname in os.listdir(template_folder):
        # Get the name of the subfolder 
        subfolder = os.path.join(input_folder, dirname)
        if not os.path.isdir(subfolder): 
          continue
        # Extract the label
        label = subfolder[subfolder.rfind('/') + 1:]
        print(label)
        # Initialize variables
        X = np.array([])
        y_words = []
    
    for filename in [x for x in os.listdir(subfolder) if x.endswith('.wav')][:-1]:
        # Read the input file
        filepath = os.path.join(subfolder, filename)
        audio, sampling_freq = librosa.load(filepath)
        # Extract MFCC features
        mfcc_features = mfcc(y=audio, sr=sampling_freq)
        # Append to the variable X
        if len(X) == 0:
          X = mfcc_features[:,:15]
        else:
          X = np.append(X, mfcc_features[:,:15], axis=0)
        # Append the label
        y_words.append(label)
    print('X.shape =', X.shape)
    
    
    # train model
    if dataset == 'learn_phase':
        #! small dataset from learning phase
        train_start = time.time()
        hmm_trainer = HMMTrainer()
        hmm_trainer.train(X)
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
      score = hmm_model.get_score(mfcc_features)
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


  '''
#! functions for execution on embedded devise


def recognize(templates):
    # defining some parameters, that were estimated empirically
    metric = cityblock
    # coef_mfcc = 0.96
    # coef_fft = 4.0000000000000036
    # reshaping_factor = 7
    
    # record voice command and extract it's features
    sample = record_process_audio()
    sample = process_signal(sample, RATE, 'MFCC')

    
    # dynamic time wrapping to pair corresponding frames
    # return a distance between two time sequences - similarity measure
    start = time.time()
    distances = [] 
    match = 'none'
    for template in templates:
        distance_mfcc, cost, acc_cost, path = dtw(template[1].T, sample[1].T, dist=metric)
        distances.append((template[0], distance_mfcc))
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


def training(keywords, commandos, mode='FM'):
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
