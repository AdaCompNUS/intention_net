import cv2
import numpy as np
import glob
from matplotlib import pyplot as plt

def undistort(img):
    # K - Intrinsic camera matrix for the raw (distorted) images.
    camera_matrix =  [714.3076782226562, 0.0, 642.54052734375, 0.0, 714.65966796875, 382.0349426269531, 0.0, 0.0, 1.0]
    camera_matrix = np.reshape(camera_matrix, (3, 3))

    # distortion parameters - (k1, k2, t1, t2, k3)
    distortion_coefs = [-0.3165473937988281, 0.1023712158203125, -1.52587890625e-05, -0.000728607177734375, 0.0]
    distortion_coefs = np.reshape(distortion_coefs, (1, 5))

    # R - Rectification matrix - stereo cameras only, so identity
    rectification_matrix =  [0.9999804496765137, -0.006237626075744629, 0.0003840923309326172, 0.006237149238586426, 0.9999796152114868, 0.001311659812927246, -0.0003923177719116211, -0.0013092756271362305, 0.9999990463256836]  
    rectification_matrix = np.reshape(rectification_matrix,(3,3))
    # P - Projection Matrix - specifies the intrinsic (camera) matrix
    #  of the processed (rectified) image
    projection_matrix =[698.4000244140625, 0.0, 649.08251953125, 0.0, 0.0, 698.4000244140625, 377.290771484375, 0.0, 0.0, 0.0, 1.0, 0.0]
    projection_matrix = np.reshape(projection_matrix, (3, 4))
    
    # Not initialized - initialize all the transformations we'll need
    mapx = np.zeros(img.shape)
    mapy = np.zeros(img.shape)

    H, W, _ = img.shape

    # Initialize self.mapx and self.mapy (updated)
    mapx, mapy = cv2.initUndistortRectifyMap(camera_matrix, 
        distortion_coefs, rectification_matrix, 
        projection_matrix, (W, H), cv2.CV_32FC1)

    return cv2.remap(img, mapx, mapy, cv2.INTER_NEAREST)
    
for img in glob.glob("left*.jpg"):
    img = cv2.imread(img)
    img = undistort(img)
    plt.imshow(img)
    plt.show()
    