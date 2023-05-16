
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
'''
# metric
# features
features = ['FFT', 'MFCC', 'FM']
metrics = [euclidean, braycurtis, canberra, chebyshev, cityblock, correlation, cosine, sqeuclidean, (lambda x, y: norm(x - y, ord=1))]
times = []

for feature in features:
    for metric in metrics:
        time = recognize_prerecorded(feature, metric)
        times.append((feature, metric, time))
        
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



#************************************************************
#********************Testing prerecorded*********************
#************************************************************


recognize_prerecorded('FM')


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