import cv2
import numpy as np
import json
from stereovision.calibration import StereoCalibration
from datetime import datetime
import open3d as o3d
import os


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

img_height = 1232
img_width = 3072

# Implementing calibration data
print('Read calibration data and rectifying stereo pair...')
calibration = StereoCalibration(input_folder='calib_result')

sbm = cv2.StereoBM_create(numDisparities=0, blockSize=21)

# mm
baseline = 60
# mm
focal = 2.6
# mm
pxl_size = 1.12 / 1000.0
pxl_size = pxl_size * 3280 / 1536

def stereo_depth_map(rectified_pair):
    dmLeft = rectified_pair[0]
    dmRight = rectified_pair[1]
    disparity = sbm.compute(dmLeft, dmRight)
        
    disparity_real = (disparity - 610) / 11

    depth = baseline * focal / (disparity_real * pxl_size) 
    depth[depth == depth.max()] = 0
    
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

imageToDisp = './scenes/photo.png'
if os.path.isfile(imageToDisp) == False:
    print ('Can not read image from file \"'+imageToDisp+'\"')
    exit(0)

frame = cv2.imread(imageToDisp)
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

t1 = datetime.now()
pair_img = cv2.cvtColor (frame, cv2.COLOR_BGR2GRAY)
imgLeft = pair_img [0:img_height,0:int(img_width/2)] #Y+H and X+W
imgRight = pair_img [0:img_height,int(img_width/2):img_width] #Y+H and X+W
rectified_pair = calibration.rectify((imgLeft, imgRight))
depth_map = stereo_depth_map(rectified_pair)

imgLeftC = frame[0:img_height,0:int(img_width/2)]
color_raw = o3d.geometry.Image((imgLeftC).astype(np.uint8))
depth_raw = o3d.geometry.Image((depth_map).astype(np.uint16))
rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw, convert_rgb_to_intensity=False)

width = 1536
height = 1232
fx = 2.6 / pxl_size
fy = fx
cx = width / 2.0
cy = height / 2.0

intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx,fy, cx, cy)

pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)

# bigger voxel_size - less amount of points
voxel_down_pcd = pcd.voxel_down_sample(voxel_size=0.02)

cl, ind = voxel_down_pcd.remove_radius_outlier(nb_points=10, radius=0.04)
inlier_pcd = voxel_down_pcd.select_by_index(ind)

cl, ind = inlier_pcd.remove_radius_outlier(nb_points=25, radius=0.08)
inlier_inlier_pcd = inlier_pcd.select_by_index(ind)

o3d.visualization.draw_geometries([inlier_inlier_pcd])
o3d.io.write_point_cloud("fragment.ply", inlier_inlier_pcd)

t2 = datetime.now()
print ("DM build time: " + str(t2-t1))


