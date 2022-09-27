import cv2
import json
import numpy as np
import open3d as o3d
from stereovision.calibration import StereoCalibration


class PCDExtractor(object):
    """Class to convert stereoimage to 3D cloud of points."""

    # Camera resolution
    CAM_W = 3280
    CAM_H = 2464

    # Camera params
    baseline = 60  # mm
    focal = 2.6  # mm
    pxl_size = 1.12 / 1000.0  # mm

    def __init__(self, stereophoto, stereo_photo_width, stereo_photo_height):
        self.stereophoto = stereophoto
        
        stereo_photo_width = (stereo_photo_width + 31) // 32 * 32
        stereo_photo_height = (stereo_photo_height + 15) // 16 * 16
        self.photo_width, self.photo_height = stereo_photo_width, stereo_photo_height
        
        self.img_width = stereo_photo_width // 2
        self.img_height = stereo_photo_height

        # Recalc pixel size for different resolution
        self.pxl_size = self.pxl_size * self.CAM_W / self.img_width

        self.calibration = StereoCalibration(input_folder='calib_result')

        self.sbm = cv2.StereoBM_create(numDisparities=0, blockSize=21)

    def load_map_settings(self, filename):
        with open(filename, 'r') as f:
            data = json.load(f)

        SWS = data['SADWindowSize']
        PFS = data['preFilterSize']
        PFC = data['preFilterCap']
        MDS = data['minDisparity']
        NOD = data['numberOfDisparities']
        TTH = data['textureThreshold']
        UR = data['uniquenessRatio']
        SR = data['speckleRange']
        SPWS = data['speckleWindowSize']   
        
        self.sbm.setPreFilterType(1)
        self.sbm.setPreFilterSize(PFS)
        self.sbm.setPreFilterCap(PFC)
        self.sbm.setMinDisparity(MDS)
        self.sbm.setNumDisparities(NOD)
        self.sbm.setTextureThreshold(TTH)
        self.sbm.setUniquenessRatio(UR)
        self.sbm.setSpeckleRange(SR)
        self.sbm.setSpeckleWindowSize(SPWS)

    def get_rectified_pair(self):
        pair_img = cv2.cvtColor(self.stereophoto, cv2.COLOR_BGR2GRAY)
        img_left = pair_img[0:self.img_height,0:self.img_width]
        img_right = pair_img[0:self.img_height,self.img_width:self.photo_width]
        self.rectified_pair = self.calibration.rectify((img_left, img_right))

    def calc_disparity(self):
        dmLeft, dmRight = self.rectified_pair
        disparity = self.sbm.compute(dmLeft, dmRight)
        # Transform original disparity to camera specific
        disparity_real = (disparity - 610) / 11
        self.disparity = disparity_real

    def calc_depth(self):
        self.depth = self.baseline * self.focal / (self.disparity * self.pxl_size)
        # Remove background depths
        self.depth[self.depth == self.depth.max()] = 0

    def calc_pointcloud(self):
        img_left = self.stereophoto[0:self.img_height,0:self.img_width]   
        color_raw = o3d.geometry.Image(img_left.astype(np.uint8))
        depth_raw = o3d.geometry.Image(self.depth.astype(np.uint16))
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw, convert_rgb_to_intensity=False)
        
        width, height = self.img_width, self.img_height
        fx = self.focal / self.pxl_size
        fy = fx
        cx, cy = width / 2.0, height / 2.0
        intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

        self.pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)
    
    def show_pointcloud(self):
        voxel_down_pcd = self.pcd.voxel_down_sample(voxel_size=0.02)

        cl, ind = voxel_down_pcd.remove_radius_outlier(nb_points=10, radius=0.04)
        inlier_pcd = voxel_down_pcd.select_by_index(ind)

        cl, ind = inlier_pcd.remove_radius_outlier(nb_points=25, radius=0.08)
        inlier_inlier_pcd = inlier_pcd.select_by_index(ind)

        o3d.visualization.draw_geometries([inlier_inlier_pcd])
        o3d.io.write_point_cloud('fragment.ply', inlier_inlier_pcd)


if __name__ == '__main__':
    stereophoto = cv2.imread('./scenes/photo.png')
    stereophoto = cv2.cvtColor(stereophoto, cv2.COLOR_BGR2RGB)

    pcd_extractor = PCDExtractor(stereophoto, 3072, 1232)
    pcd_extractor.load_map_settings ('3dmap_set.txt')
    pcd_extractor.get_rectified_pair()
    pcd_extractor.calc_disparity()
    pcd_extractor.calc_depth()
    pcd_extractor.calc_pointcloud()
    pcd_extractor.show_pointcloud()
