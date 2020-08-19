import Jetson.GPIO as GPIO
import time
from threading import Thread

def checkSuccess(target_time, output_pin):
  global success
  start=time.time()
  while True:
    current_time=time.time()
    if (current_time-start>=target_time):
      GPIO.output(output_pin,GPIO.LOW)
      success = True
      # time.sleep(1)
      return 0
    # print(success)
################# Initialize Various ###############
RT_pin=18	#turn right pin
LT_pin=13	#turn left pin  
RB_pin=0	#vibration right pin
LB_pin=0	#vibration left pin
GPIO.setmode(GPIO.BCM)
GPIO.setup(RT_pin,GPIO.OUT)
GPIO.setup(LT_pin,GPIO.OUT)
GPIO.setup(RB_pin,GPIO.OUT)
GPIO.setup(LB_pin,GPIO.OUT)

TA=0 #target_angle
PA=0 #previous_angle
PTA=0 #previous_target_angle
p_t = 0 #previous_time
success = False
direction = 0

# TT #target_time
# PT #previous_time

#################### Main Loop ####################
theta = 73
for frame in range(10):

  TA = theta*9
  print(success)
  if frame==0: pass
  else: 
    t.do_run = False

    
  c_t=time.time()
  if success==False: CA=(c_t-p_t)*216*direction+PA	
  else: CA=PTA

  if TA == CA: pass
  else:
    direction = (TA-CA)/abs(TA-CA)
    TT = abs(TA-CA)/216
    if direction<0:
      GPIO.output(RT_pin,GPIO.HIGH)
      output_pin=RT_pin
    else:
      GPIO.output(LT_pin,GPIO.HIGH)
      output_pin=LT_pin
    t=Thread(target=checkSuccess, args=(TT,output_pin))
    t.start()
  p_t=time.time()
  PTA=TA
  PA=CA
  time.sleep(0.5)
