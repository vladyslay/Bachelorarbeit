# helper funktions used by on_off_set.py and speech_recognition.py
import numpy as np
from scipy.fftpack import fft
from scipy import signal
import audioop 
import pyaudio
import math
from scipy import signal
from scipy.spatial.distance import euclidean
import numpy as np

f1_threshhold = 0.4
f2_threshhold = 30
f3_threshhold = 0.9

FRAME = 10      #ms
RATE = 44100    #Hz
#CHUNK = int(RATE * FRAME * pow(10, -3))
CHUNK = 4096
FORMAT = pyaudio.paInt16
CHANNELS = 1

# Generate reference frequency for 
x = np.linspace(0,RATE,RATE)
#ref = 300*np.sin(10000*2*np.pi*x)   #250 in freq.py
ref = 250*np.sin(10000*2*np.pi*x)
ref = ref[0:CHUNK]

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



def extractFeatures(fft_norm, hf_range, lf_range,vol_tres, mode="on-off-set"):
	if mode == "speech_recognition":
		#TODO 
		a = 0
  
	# mode == on-off-set
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


def dtw_table(x, y, distance=None):
    if distance is None:
        distance = euclidean
    nx = len(x)
    ny = len(y)
    table = np.zeros((nx+1, ny+1))
    
    #compute left column separately, j=0
    table[1:, 0] = np.inf
    
    #compute top row separately, i=0
    table[0, 1:] = np.inf
    
    for i in range(1, nx+1):
        for j in range(1, ny+1):
            d = distance(x[i-1], y[j-1])
            table[i, j] = d + min(table[i-1, j], table[i, j-1], table[i-1,j-1])
    return table

def dtw(x, y, table):
    i = len(x)
    j = len(y)
    path = [(i,j)]
    while i > 0 or j > 0:
        minval = np.inf
        if table[i-1][j-1] < minval:
            minval = table[i-1][j-1]
            step = (i-1, j-1)
        if table[i-1][j] < minval:
            minval = table[i-1][j]
            step = (i-1, j)
        if table[i][j-1] < minval:
            minval = table[i][j-1]
            step = (i, j-1)
        path.insert(0, step)
        i,j = step
    return np.array(path)


