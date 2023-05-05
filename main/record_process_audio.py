
import numpy as np
from scipy.fftpack import fft
from scipy import signal
import sys
import time
import pyaudio
import matplotlib.pyplot as plt
import pigpio as GPIO           
from helperf import *

pause_timer = 4 #! magic number

#start = time.time()         #[10.2]
#print("Start")


# ******************************************************
# ************* Define params **************************
# ******************************************************
'''
FRAME = 10      #ms
RATE = 44100    #Hz
#CHUNK = int(RATE * FRAME * pow(10, -3))
CHUNK = 4096
FORMAT = pyaudio.paInt16
CHANNELS = 1
'''


audio = pyaudio.PyAudio()

stream = audio.open(
    format = FORMAT,
    channels = CHANNELS,
    rate = RATE,
    input_device_index = 1,
    input = True,
    frames_per_buffer = CHUNK
)

# Datastructures
#TODO define structures for templates and utteraces
templates = {}	# dict to store transcription alongside of a piece of audio in form "sound" : np.array
# utterance = [np.array()] array von np arrays

 

# ******************************************************
# ************* Main Script ****************************
# ******************************************************

#Bereiche: [low,high] --> what are these ? 
hf_range = [5000,9000]
lf_range = [0,10]


# Calc filter coefficients 
hp_coeff = signal.bessel(2, 100, btype='highpass', output='sos', fs=RATE)
lp_coeff = signal.bessel(10, 11000, btype='lowpass', output='sos', fs=RATE)
win_han = signal.windows.hann(CHUNK)


#print("start") # show where onset detection loop starts 

#  Main loop for running over the audio signal 

onset_detected = False
onset_pause = False
offset_detected = False
speech = True
d = 0

pause = 0
n = 0 
end = 0
st_tres =[] 


#TODO make a funktion return recorded signal
# funktion returns a chunk of audio between on/off-set pair
def record_process_audio():
    # here recorded speech is stored
    speech_piece = np.empty()
    speech_recorded = False
    
    while(speech_recorded == False):
        a = 0
        # start recording audio
        data_raw =  stream.read(CHUNK)

        #? if speech is not present anymore - set endpoint
        if (speech == False):
            end = time.time() - pause
            if end >= pause_timer:
                speech = True
                offset_detected = False
        else: # speech = True
            vol_tres = volumeFilter(data_raw)
            data_buffer_int = np.frombuffer(data_raw, dtype=np.int16)
            data_int = np.add(data_buffer_int, ref) 
            # print("Amplitude:  ", np.mean(data_buffer_int)), #andere art zum ausgeben der amplitude

            #! filtered data
            filtered_data = applyFilter(data_int, win_han, lp_coeff, hp_coeff)

            # apply fft (tranform to frequency spectrum)
            fft_norm = applyFFT(filtered_data, RATE, win_han, lp_coeff, hp_coeff)
            # calc number of features extracted
            num_features_detected = extractFeatures(fft_norm, hf_range, lf_range,vol_tres)

            # check if speech onset is detected
            if num_features_detected >= 4 and offset_detected == False: #zusaetzliches feature lautstaerke
            
                if d < 1:
                    print("onset")
                    time.sleep(0.005)
                    onset_detected = True
                d += 1
            elif onset_detected == True and d ==1 and num_features_detected >= 3:
                d += 1	# pass if after onset only 3 features detected
            elif onset_detected == True and d >= 2:
                print("offset")
                time.sleep(0.005)
                onset_detected = False
                offset_detected = True
                speech = False
                speech_recorded = True
                pause = time.time() 
                d = 0
            elif onset_detected == True and d == 1:
                print("Stoergeraeusch erkannt")
                time.sleep(0.0001)
                d = 0
                onset_detected = False
                offset_detected = False
    return speech_piece

#TODO get rid of the pause after offset
# code below is commented
'''
# recording audio and setting offsets
while (True):
    try:
        # start recording audio
        data_raw =  stream.read(CHUNK)
        #? if speech is not present anymore - set endpoint
        if (speech == False):
            end = time.time() - pause
            if end >= pause_timer:
                speech = True
                offset_detected = False
        else: # speech = True
            vol_tres = volumeFilter(data_raw)
            data_buffer_int = np.frombuffer(data_raw, dtype=np.int16)
            data_int = np.add(data_buffer_int, ref) 
            # print("Amplitude:  ", np.mean(data_buffer_int)), #andere art zum ausgeben der amplitude
            
            #! filtered data
            filtered_data = applyFilter(data_int, win_han, lp_coeff, hp_coeff)
            
            # apply fft (tranform to frequency spectrum)
            fft_norm = applyFFT(filtered_data, RATE, win_han, lp_coeff, hp_coeff)
            # calc number of features extracted
            num_features_detected = extractFeatures(fft_norm, hf_range, lf_range,vol_tres)
         
            # check if speech onset is detected
            if num_features_detected >= 4 and offset_detected == False: #zusaetzliches feature lautstaerke
            
                if d < 1:
                    print("onset")
                    time.sleep(0.005)
                    onset_detected = True
                d += 1
            elif onset_detected == True and d ==1 and num_features_detected >= 3:
                d += 1	# pass if after onset only 3 features detected
            elif onset_detected == True and d >= 2:
                print("offset")
                time.sleep(0.005)
                onset_detected = False
                offset_detected = True
                speech = False
                pause = time.time() 
                d = 0
            elif onset_detected == True and d == 1:
                print("Stoergeraeusch erkannt")
                time.sleep(0.0001)
                d = 0
                onset_detected = False
                offset_detected = False
                
                
    except(KeyboardInterrupt):
        print("time.time() - start", time.time() - start)
        print("end", end)
        
        time.sleep(0.001)
        stream.stop_stream()
        stream.close()
        
        #times = np.linspace(0, end, num=CHUNK)
        #plt.plot(times, filtered_data)
        #plt.show()
        
        
        
        sys.exit()
'''