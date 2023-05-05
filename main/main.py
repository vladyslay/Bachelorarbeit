
#************************************************************
#********************Initialisierung*************************
#************************************************************
#import numpy as np
#from scipy.fftpack import fft
#from scipy import signal
#import sys
import time
#import pyaudio
#import math
#import audioop 
#import matplotlib.pyplot as plt
from speech_recognition import recognize, training, recognize_prerecorded
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



#************************************************************
#********************Testing prerecorded*********************
#************************************************************


recognize_prerecorded('MFCC')


'''

#************************************************************
#************************Learning phase**********************
#************************************************************

# define the list of needed keywords and commands
keywords = ["exo", "ok", "nein"]
commandos = []

training_finished = False
templates_keywords, templates_commandos = training(keywords, commandos)

#************************************************************
#********************Recognition phase***********************
#************************************************************

keyword_recognized = False
commando_recognized = False

while keyword_recognized == False:
    # has to return a val for keyword_recognize
    keyword_recognized, matched_keyword = recognize(keywords)
    while commando_recognized == False and keyword_recognized == True:
        # has to return a val for commando_recognized 
        commando_recognized, matched_commando = recognize(commandos)



#************************************************************
#********************Feedback phase**************************
#************************************************************

feedback_recognized = False
recognize()

'''