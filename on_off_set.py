
import numpy as np
from scipy.fftpack import fft
from scipy import signal
import sys
import time
import pyaudio
import math
import audioop 
import matplotlib.pyplot as plt
########################################## Ansprechen des Boards 
#GPIO:                                          #[3]
# import RPi.GPIO as GPIO 
import pigpio as GPIO           

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
################ 

# ******************************************************
# ************* Functions ******************************
# ******************************************************
start = time.time()         #[10.2]
print("Start")

# to check git

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





def extractFeatures(fft_norm, hf_range, lf_range,vol_tres):

	features = 0
	# Peak von Referenzfrequenz:d_pcm_usb_stream_open) Invalid type for card

	# Relativ guter Grundindikator. Wenn Sprache vorhanden, auf jeden Fall positives Ergebnis
	if fft_norm[10000] < f1_threshhold: #< 0.25:  #0,4 in freq.py ?????????
		features += 1
			
	# Frequenzen im Bereich von 5000 bis 9000 Hz vorhanden?
	# Oft Ausschlaege bei herunterfallenden Gegenstaenden, klatschen etc.
	if np.sum(fft_norm[hf_range[0]:hf_range[1]]) < f2_threshhold:  
		features += 1

	# Frequenzen im Bereich von 0 bis 100 Hz?
	# Bei Sprache weniger vorhanden, gut um dumpfe Geraeusche auszusortieren
	if np.sum(fft_norm[lf_range[0]:lf_range[1]]) < f3_threshhold: #< 1.5: #0,9 in freq.py
		features += 1

	if vol_tres >= 30:
		features += 1

	return features # number of features detected 




# ******************************************************
# ************* Define params ****************************
# ******************************************************

FRAME = 10      #ms
RATE = 44100    #Hz
#CHUNK = int(RATE * FRAME * pow(10, -3))
CHUNK = 4096
FORMAT = pyaudio.paInt16
CHANNELS = 1

f1_threshhold = 0.4
f2_threshhold = 30
f3_threshhold = 0.9

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



# Generate reference frequency for 
x = np.linspace(0,RATE,RATE)
#ref = 300*np.sin(10000*2*np.pi*x)   #250 in freq.py
ref = 250*np.sin(10000*2*np.pi*x)
ref = ref[0:CHUNK]


print("start") # show where onset detection loop starts 


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


#old loop here
'''
while(True):
	try:
		
		#start recording audio signal
		data_raw = stream.read(CHUNK)
		
		if speech == False:
			end = time.time() - pause
			if end >= pause_timer:
				speech = True
				offset_detected = False
			#print(end), zum anzeigen der 4 sekunden
		else:
			vol_tres = volumeFilter(data_raw)
			data_buffer_int = np.frombuffer(data_raw, dtype=np.int16)
			data_int = np.add(data_buffer_int, ref)
			#print("Amplitude:  ", np.mean(data_buffer_int)), andere art zum ausgeben der amplitude
			
			#! filter data 
			filtered_data = applyFilter(data_int, win_han, lp_coeff, hp_coeff)

			# apply fft (tranform to frequency spectrum)
			fft_norm = applyFFT(filtered_data, RATE, win_han, lp_coeff, hp_coeff)
			# calc number of features extracted 
			num_features_detected = extractFeatures(fft_norm, hf_range, lf_range,vol_tres)

			

			
			# looking for the end of speech utterance
			# Check if speech onset is detected    ##################### ab hier geaendert 
			if num_features_detected >= 4 and offset_detected == False: #zusaetzliches feature lautstaerke 
				if (d < 1):
					print("onset")
					#GPIO.output(pin_out, 1)         #[3]
					time.sleep(0.005)              #[10.1]
					#GPIO.output(pin_out, 0)
					onset_detected = True
					

				d += 1 
			elif (onset_detected == True and d == 1 and num_features_detected >= 3):	
				d += 1	# pass if after onset only 3 features detected 			
			elif (onset_detected == True and d >= 2):
				print("offset")
				#GPIO.output(pin_off, 1)
				time.sleep(0.005)
				#GPIO.output(pin_off, 0)
				onset_detected = False
				offset_detected = True
				speech = False
				pause = time.time()
				d = 0


			elif(onset_detected == True and d == 1):	#stoergeraeusche auch beim wort erkannt
				print("Stoergeraeusch erkannt")
				#GPIO.output(pin_stoer, 1)         #[3]
				time.sleep(0.0001)              #[10.1]
				#GPIO.output(pin_stoer, 0)
				d = 0
				onset_detected = False
				offset_detected = False	
	except KeyboardInterrupt:
		print(time.time()-start)
		#GPIO.output(pin_out, 1)
		time.sleep(0.001)
		#GPIO.output(pin_out,0)
		stream.stop_stream()    #[1]
		stream.close()          #[1]
		
		#GPIO.cleanup()          #[3]
		sys.exit()              #[11]		    

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