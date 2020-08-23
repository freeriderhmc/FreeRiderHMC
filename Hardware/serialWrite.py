# pyserial
import numpy as np
import serial
import time
import math

ser = serial.Serial("/dev/ttyUSB1")
ser.baudrate = 115200
frame_num = 0
target_time = 0
target_angle = 5.36
eol = '\n'

while 1:
    
    target_angle = -target_angle
    
    if frame_num % 3 == 0:
        invade = "l"
    elif frame_num % 3 == 1:
        invade = "r"
    elif frame_num % 3 == 2:
        invade = "n"

    time.sleep(0.5)

    frame_num_write = "0000" + str(frame_num)
    frame_num_write = frame_num_write[-5:]

    target_angle = round(target_angle, 5)
    target_angle_write = "0000000000" + str(target_angle)
    target_angle_write = target_angle_write[-10:]
    print(len(str(target_angle)))
    target_angle_len = str(len(str(target_angle)))
    
    message = frame_num_write + target_angle_write + invade + target_angle_len
    # frame_num_write = "frame" + str(frame_num)
    # target_angle_write = str(target_angle)
    # ser.write(frame_num_write.encode("cp949"))
    # ser.write(eol.encode("cp949"))
    # time.sleep(0.001)

    # ser.write(target_angle_write.encode("cp949"))
    # ser.write(eol.encode("cp949"))
    # time.sleep(0.001)

    ser.write(message.encode("cp949"))
    ser.write(eol.encode("cp949"))
    time.sleep(0.001)

    frame_num += 1



