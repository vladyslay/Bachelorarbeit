
#************************************************************
#********************Initialisierung*************************
#************************************************************
#import sys
import time
from scipy.spatial.distance import *
from numpy.linalg import norm
from speech_recognition import recognize, training, recognize_prerecorded, recognize_prerecorded_ml
from speech_recognition_ml import *
########################################## Ansprechen des Boards 
#GPIO:                                          #[3]
# import RPi.GPIO as GPIO 
#import pigpio as GPIO           

#pin_out = 37
#pin_stoer = 38  
#pin_detect = 36
#pin_off = 35

pause_timer = 4 #! magic number

#GPIO.setmode(GPIO.BOARD)            
#GPIO.setup(pin_out, GPIO.OUT)       
#GPIO.setup(pin_stoer, GPIO.OUT)
#GPIO.setup(pin_detect, GPIO.IN)
#GPIO.setup(pin_off, GPIO.OUT)

mode_mfcc = 'MFCC'
mode_fft = 'FFT'
mode_mfcc_fft = 'FM'

#************************************************************
#**************figuring out best parameters******************
#************************************************************

# metric
# features
# global constraint
features = ['FFT', 'MFCC', 'FM']
metrics = [euclidean, braycurtis, 
           canberra, chebyshev, 
           cityblock, correlation, 
           cosine, sqeuclidean, 
           (lambda x, y: norm(x - y, ord=1))]

configurations = {}

for metric in metrics:
    print('Using:', metric)
    matching_time, correctness = recognize_prerecorded(mode_mfcc_fft, metric, 0.96, 4.0000000000000036, 7)
    #matching_time = sum(matching_time)/len(matching_time)
    configurations.update({metric: correctness})
print('Configurations:')
print(configurations)
best_config = max(configurations, key=configurations.get)
print('best config:', best_config)
    
    
    
'''
for feature in features:
    for metric in metrics:
        print('Using:', feature, metric)
        matching_time, correctness = recognize_prerecorded(feature, metric)
        matching_time = sum(matching_time)/len(matching_time)
        configurations.append((feature, metric, matching_time, correctness))
print('Configurations:')
print('Feature | Metric | Matching time | Correctness')
for i in range(1, len(metrics)):
    print(configurations[i][0], configurations[i][1], configurations[i][2], configurations[i][3])

'''
# according to above compare the best metric for mfcc is cityblock
#! best_metric = cityblock

'''
reshaping_factors = {}
matching_time = 0
for i in range(1, 20):
    if 22050 % i == 0:
        matching_time, correctness = recognize_prerecorded(mode_mfcc_fft, best_metric, i * 0.1, (1 - i * 0.1) * 100, i) 
        reshaping_factors.update({i: correctness})    
print('Reshaping faktors:')
print('Faktor | Correctness')
print(reshaping_factors)
best_corectness = max(reshaping_factors, key=reshaping_factors.get)
print('reshaping factor with best performance:', best_corectness)
'''
#! according to the bove code the best reshaping factor for fft features is 7
    

'''
# try if combination of MFCC with FFT with different weights 
# has better matching results
coefficients = {}
for i in range(1, 100):
    matching_time, correctness = recognize_prerecorded(mode_mfcc_fft, best_metric, i * 0.01, (1 - i * 0.01) * 1000) 
    matching_time = sum(matching_time)/len(matching_time)
    coefficients.update({( i * 0.01, (1 - i * 0.01) * 100): correctness})
print('Coefficients:')
print('Feature | Metric | Coef. MFCC | Cef. FFT | Matching time | Correctness')
print(coefficients)
best_coefs = max(coefficients, key=coefficients.get)
print('best coefs:', best_coefs)
#! best coefs: (0.96, 4.0000000000000036)
'''

'''
#! HMM and DTW compare
matching_time_tm, correctness_tm = recognize_prerecorded(mode_mfcc)
correctness_ml, training_time, matching_time_ml = recognize_prerecorded_ml(mode_mfcc, 'learn_phase')

print('Template matching')
print("Matching times | Correctness")
for i in range [1, len(matching_time_tm)]:
    print(matching_time_tm[i], correctness_tm[i])

print('HMM')
print("Matching times | Correctness")    
for i in range [1, len(matching_time_tm)]:
    print(matching_time_ml[i], correctness_ml[i])
print('Training time:', training_time)



'''

#************************************************************
#********************Testing prerecorded*********************
#************************************************************


#recognize_prerecorded(mode_fft, minkowski)


'''

#************************************************************
#************************Learning phase**********************
#************************************************************

# define the list of needed keywords and commands
keywords = ["exo", "ok", "nein"]
#TODO develope commandos
commandos = []

# training process itself, returns sets of templates
templates_keywords, templates_commandos = training(keywords, commandos)

#************************************************************
#********************Recognition phase***********************
#************************************************************

keyword_recognized = False
commando_recognized = False

# start with keyword recognition
matched_keyword, matching_keyword_time = recognize(keywords, 'MFCC')
if matched_keyword == "exo":
    # match commando
    matched_commando, matching_commando_time = recognize(commandos, 'MFCC')
    
    #************************************************************
    #********************Feedback phase**************************
    #************************************************************

    matched_feedback, matching_commando_time = recognize(keywords, 'MFCC')
    
    if matched_feedback == 'ok':
        # output interface
        forward_command(matched_commando)
    else:
        


'''