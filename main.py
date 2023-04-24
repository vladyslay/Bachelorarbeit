
#************************************************************
#********************Initialisierung*************************
#************************************************************
import numpy as np
from scipy.fftpack import fft
from scipy import signal
import sys
import time
import pyaudio
import math
import audioop 
import matplotlib.pyplot as plt
from speech_recognition import recognize_commando, recognize_feedback, recognize_keyword, training
from on_off_set import on_off_set
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








#************************************************************
#************************Learning phase**********************
#************************************************************



#************************************************************
#********************Recognition phase***********************
#************************************************************




#************************************************************
#********************Feedback phase**************************
#************************************************************