import cv2
import numpy as np
import json
from stereovision.calibration import StereoCalibration
from datetime import datetime


# Depth map default preset
SWS = 5
PFS = 5
PFC = 29
MDS = -30
NOD = 160
TTH = 100
UR = 10
SR = 14
SPWS = 100

stream = cv2.VideoCapture('http://192.168.97.104:8000/stream.mjpg')

img_height = 1232
img_width = 3072

# Implementing calibration data
print('Read calibration data and rectifying stereo pair...')
calibration = StereoCalibration(input_folder='calib_result')

sbm = cv2.StereoBM_create(numDisparities=0, blockSize=21)

def stereo_disparity_map(rectified_pair):
    dmLeft = rectified_pair[0]
    dmRight = rectified_pair[1]
    disparity = sbm.compute(dmLeft, dmRight)
    
    disparity_normalized = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
    disparity_visual = np.array(disparity_normalized, dtype = np.uint8)
    disparity_color = cv2.applyColorMap(disparity_visual, cv2.COLORMAP_JET)
    disparity_colorS = cv2.resize(disparity_color, (800, 540))
    cv2.imshow("Image", disparity_colorS)
    key = cv2.waitKey(1) & 0xFF   
    if key == ord("q"):
        quit()
    return disparity_color

def load_map_settings( fName ):
    global SWS, PFS, PFC, MDS, NOD, TTH, UR, SR, SPWS, loading_settings
    print('Loading parameters from file...')
    f=open(fName, 'r')
    data = json.load(f)
    SWS=data['SADWindowSize']
    PFS=data['preFilterSize']
    PFC=data['preFilterCap']
    MDS=data['minDisparity']
    NOD=data['numberOfDisparities']
    TTH=data['textureThreshold']
    UR=data['uniquenessRatio']
    SR=data['speckleRange']
    SPWS=data['speckleWindowSize']    
    #sbm.setSADWindowSize(SWS)
    sbm.setPreFilterType(1)
    sbm.setPreFilterSize(PFS)
    sbm.setPreFilterCap(PFC)
    sbm.setMinDisparity(MDS)
    sbm.setNumDisparities(NOD)
    sbm.setTextureThreshold(TTH)
    sbm.setUniquenessRatio(UR)
    sbm.setSpeckleRange(SR)
    sbm.setSpeckleWindowSize(SPWS)
    f.close()
    print ('Parameters loaded from file '+fName)

load_map_settings ("3dmap_set.txt")

while(True):
    # Capture the video frame by frame
    _, frame = stream.read()

    t1 = datetime.now()
    pair_img = cv2.cvtColor (frame, cv2.COLOR_BGR2GRAY)
    imgLeft = pair_img [0:img_height,0:int(img_width/2)] #Y+H and X+W
    imgRight = pair_img [0:img_height,int(img_width/2):img_width] #Y+H and X+W
    rectified_pair = calibration.rectify((imgLeft, imgRight))
    disparity = stereo_disparity_map(rectified_pair)

    imgRightS = cv2.resize(imgRight, (480, 540)) 
 
    # show the frame
    cv2.imshow("left", imgRightS)    

    t2 = datetime.now()
    print ("DM build time: " + str(t2-t1))
