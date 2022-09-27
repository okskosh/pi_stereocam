import cv2
import time
from datetime import datetime


filename = './scenes/photo.png'
stream = cv2.VideoCapture('http://192.168.159.104:8000/stream.mjpg')

total_photos = 30             # Number of images to take
countdown = 5                 # Interval for count-down timer, seconds
font=cv2.FONT_HERSHEY_SIMPLEX # Cowntdown timer font

t2 = datetime.now()
counter = 0
avgtime = 0

while(True):
    # Capture the video frame by frame
    _, frame = stream.read()

    t1 = datetime.now()
    cntdwn_timer = countdown - int ((t1-t2).total_seconds())
    # If cowntdown is zero - let's record next image
    if cntdwn_timer == -1:
      counter += 1
      filename = './scenes/scene_' + str(counter) + '.png'
      cv2.imwrite(filename, frame)
      print (' ['+str(counter)+' of '+str(total_photos)+'] '+filename)
      t2 = datetime.now()
      time.sleep(1)
      cntdwn_timer = 0      # To avoid "-1" timer display 
      next
    # Draw cowntdown counter, seconds
    cv2.putText(frame, str(cntdwn_timer), (50,50), font, 2.0, (0,0,255),4, cv2.LINE_AA)
    
    img_sized = cv2.resize(frame, (960, 540)) 
    cv2.imshow("Take chess photo!", img_sized)
    key = cv2.waitKey(1) & 0xFF
    
    # Press 'Q' key to quit, or wait till all photos are taken
    if (key == ord("q")) | (counter == total_photos):
      break