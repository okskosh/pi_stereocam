import cv2
import numpy as np
import json
from stereovision.calibration import StereoCalibration
from datetime import datetime
import matplotlib.pyplot as plt


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

baseline = 60
focal = 2.6
pxl_size = 1.12 / 1000.0

def stereo_depth_map(rectified_pair):
    dmLeft = rectified_pair[0]
    dmRight = rectified_pair[1]
    disparity = sbm.compute(dmLeft, dmRight)

    # from_cv2 = [
    #     disparity[301][1134],
    #     disparity[72][1253],
    #     disparity[212][1375],
    #     disparity[261][941],
    #     disparity[588][927],
    #     disparity[350][1255],
    #     disparity[373][894],
    #     disparity[666][485],
    #     disparity[364][415],
    #     disparity[1138][880],
    #     disparity[372][302],
    #     disparity[605][1433],
    #     disparity[656][1435],
    #     disparity[330][1494],
    # ]
    # from_gimp = [
    #     200,
    #     205,
    #     207,
    #     120,
    #     110,
    #     204,
    #     117,
    #     135,
    #     50,
    #     140,
    #     61,
    #     180,
    #     160,
    #     225,
    # ]
    # plt.scatter(from_gimp, from_cv2)
    # plt.show()

    disparity_real = (disparity - 610) / 11
    depth = baseline * focal / (disparity_real * pxl_size)

    depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
    depth_visual = np.array(depth_normalized, dtype = np.uint8)

    depth_visual = cv2.bitwise_not(depth_visual)

    depth_color = cv2.applyColorMap(depth_visual, cv2.COLORMAP_JET)
    depth_colorS = cv2.resize(depth_color, (800, 540)) 
    cv2.imshow("Image", depth_colorS)
    key = cv2.waitKey(1) & 0xFF   
    if key == ord("q"):
        quit()
    return depth    

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
    depth = stereo_depth_map(rectified_pair)

    imgRightS = cv2.resize(imgLeft, (480, 540)) 
    # show the frame
    cv2.imshow("right", imgRightS)    

    t2 = datetime.now()
    print ("DM build time: " + str(t2-t1))
