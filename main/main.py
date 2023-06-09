
#************************************************************
#********************Initialisierung*************************
#************************************************************
#import sys
import time
from scipy.spatial.distance import *
from speech_recognition import recognize, training, recognize_prerecorded
from speech_recognition_ml import *
from helperf import *
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
#************************Learning phase**********************
#************************************************************

# define the list of needed keywords and commands
keywords = ["exo", "ok", "no"]
#TODO develope commandos
commandos = ['move', 'left', 'right', 'stop']

# training process itself, returns sets of templates
templates_keywords, templates_commandos = training(keywords, commandos)

#************************************************************
#********************Recognition phase***********************
#************************************************************


# start with keyword recognition
matched_keyword, matching_keyword_time = recognize(keywords)
if matched_keyword == "exo":
    # match commando
    matched_commando, matching_commando_time = recognize(commandos)
    
    #************************************************************
    #********************Feedback phase**************************
    #************************************************************
    print('Recognized commando:', matched_commando)
    print('Was your commando recognized right? [ok/no]')
    matched_feedback, matching_commando_time = recognize(keywords)
    
    if matched_feedback == 'ok':
        # output interface
        forward_command(matched_commando)
    else:
        print('Commando discarded')




