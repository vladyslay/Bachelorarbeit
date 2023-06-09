#import sys
import time
from scipy.spatial.distance import *
from speech_recognition import recognize, training, recognize_prerecorded
from speech_recognition_ml import *
from helperf import *
from scipy.signal.windows import *
import os
import librosa
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
mode_lpc = 'LPC'
mode_lpc_mfcc = 'LM'
mode_lpc_fft = 'LF'
mode_lpc_mfcc_fft = 'LMF'


#************************************************************
#**************figuring out best parameters******************
#************************************************************

# metric
# features
# global constraint
features = [mode_fft, mode_lpc, mode_mfcc, mode_mfcc_fft, mode_lpc_mfcc]
windows = [hann, barthann, bartlett, blackman, blackmanharris, bohman, boxcar,
           exponential, flattop, hamming, lanczos,
           parzen, taylor, triang, tukey]
metrics = [euclidean, braycurtis, 
           canberra, chebyshev, 
           cityblock, correlation, 
           cosine, sqeuclidean]



#! finding the best set of parameters for dtw

def reshaping_fft():
    reshaping_factors = {}
    matching_time = 0
    for i in range(1, 100):
        if 22050 % i == 0:
            matching_time, correctness = recognize_prerecorded(mode_fft, euclidean, reshaping_factor=i) 
            reshaping_factors.update({i: (correctness, matching_time)})    
    print('Reshaping faktors:')
    print('Faktor | Correctness')
    for item in reshaping_factors:
        print(item, reshaping_factors[item])
    #print(reshaping_factors)
    best_corectness = max(reshaping_factors, key=reshaping_factors.get)
    print('reshaping factor with best performance:', best_corectness)
    return


#! according to the bove code the best reshaping factor for fft features is 7
def combine_mfcc_fft():
    # try if combination of MFCC with FFT with different weights 
    # has better matching results
    coefficients = {}
    for i in range(1, 100, 5):
        matching_time, correctness = recognize_prerecorded(mode_mfcc_fft, euclidean, i * 0.01, (1 - i * 0.01)) 

        coefficients.update({( i * 0.01, (1 - i * 0.01)): correctness})
    print('Coefficients:')
    print('Feature | Metric | Coef. MFCC | Cef. FFT | Matching time | Correctness')
    print(coefficients)
    best_coefs = max(coefficients, key=coefficients.get)
    print('best coefs:', best_coefs)
    #! best coefs: (0.51, 0.49)
    return


def best_metric_lpc():
    configurations = {}
    for metric in metrics:
            print('Using:', metric)
            matching_time, correctness = recognize_prerecorded(mode_lpc, metric=metric, lpc_order=50)
            configurations.update({metric: (correctness, matching_time)})
    print(configurations)
    best_config = max(configurations, key=configurations.get)
    print('best config:', best_config)
    return



def lpc_order_test():
    # LPC order test

    #matching_time, correctness = recognize_prerecorded(mode_lpc, braycurtis, lpc_order=55)

    orders = {}
    for i in range(1, 16):
        matching_time, correctness = recognize_prerecorded(mode_lpc, euclidean, lpc_order=i, data='numbers')
        orders.update({i: (correctness, matching_time)})
    print('Filter orders for LPC')
    print('Order | Correctness | Matching Time')
    for item in orders:
        print(item, orders[item])
    #print(reshaping_factors)
    best_corectness = max(orders, key=orders.get)
    print('reshaping factor with best performance:', best_corectness)
    return





def weights_mfcc_lpc_fft():
    # try if combination of MFCC with LPC and FFT with different weights 
    # has better matching results
    coefficients = {}
    for i in range(1, 100, 5):
        for j in range(1, 100 - i, 5):
            matching_time, correctness = recognize_prerecorded(mode_lpc_mfcc_fft, coef1= (i * 0.01), coef2= (j * 0.01), coef3=(1 - i * 0.01 - j  * 0.01)) 
            coefficients.update({( i * 0.01, (1 - i * 0.01), (1 - i * 0.01 - j  * 0.01)): correctness})
    print('Coefficients:')
    print('Feature | Metric | Coef. MFCC | Coef. FFT | Coef. LPC | Matching time | Correctness')
    print(coefficients)
    best_coefs = max(coefficients, key=coefficients.get)
    print('best coefs:', best_coefs)
    return


def test_windows(mode):
    win_results = {}
    for window in windows:
        print('Using:', window)
        matchig_time, correctness = recognize_prerecorded(mode=mode, metric=cityblock, window=window)
        win_results.update({window: (correctness, matchig_time)})
    print('Windows:')
    for item in win_results:
        print(item, win_results[item])
    best_correctness = max(win_results, key=win_results.get)
    print('best window:', best_correctness)
    #! best window MFCC: blackman
    #! best window fft: barthann
    #! best window LPC: taylor
    return

#test_windows(mode_lpc)

# testing numbers dataset

'''

'''

matching_time, correctness = recognize_prerecorded(mode=mode_mfcc, metric=cityblock,
                                                   window=blackman, data='fruits')
print('Matching time:', matching_time)
print('Correctness:', correctness)


'''
for feature in features:    
    for metric in metrics:
        print('Using:', feature, metric)
        matching_time, correctness = recognize_prerecorded(feature, metric=metric, 
                                                           coef_mfcc=0.51, coef_fft=0.49, reshaping_factor=14)
        configurations.update({(metric, feature): (correctness, matching_time)})
print('Configurations:')    
print(configurations)
best_config = max(configurations, key=configurations.get)
print('best config:', best_config)
'''

# according to above compare the best metric for mfcc is cityblock
#! best_metric = cityblock   
    
    
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




'''
#! HMM and DTW compare
matching_time_tm, correctness_tm = recognize_prerecorded(mode_mfcc_fft, cityblock, 
                                                         0.96, 4.0000000000000036, 7)
correctness_ml, training_time, matching_time_ml = recognize_prerecorded_ml(mode_mfcc_fft, 'learn_phase')

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
