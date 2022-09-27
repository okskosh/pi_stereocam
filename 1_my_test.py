import cv2

import numpy as np
from datetime import datetime


filename = './scenes/photo.png'
stream = cv2.VideoCapture('http://192.168.97.104:8000/stream.mjpg')

t2 = datetime.now()
counter = 0
avgtime = 0

while(True):
    # Capture the video frame by frame
    _, frame = stream.read()

    counter += 1
    t1 = datetime.now()
    timediff = t1-t2
    avgtime = avgtime + (timediff.total_seconds())
    img_sized = cv2.resize(frame, (960, 540)) 
    cv2.imshow("Take a photo!", img_sized)
    key = cv2.waitKey(1) & 0xFF
    t2 = datetime.now()
    # if the `q` key was pressed, break from the loop and save last image
    if key == ord("q") :
        avgtime = avgtime/counter
        print ("Average time between frames: " + str(avgtime))
        print ("Average FPS: " + str(1/avgtime))
        cv2.imwrite(filename, frame)
        break