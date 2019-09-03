import sys, time
import numpy as np
from scipy.ndimage import filters
from cv_bridge import CvBridge
import cv2
import roslib
import rospy
from sensor_msgs.msg import CompressedImage

class image_feature:
    def __init__(self):
        '''Initialize ros publisher, ros subscriber'''
        self.subscriber = rospy.Subscriber("/train/mynteye/left_img/compressed",
                                    CompressedImage, self.callback,queue_size=1,buff_size=2**10)
        self.last = None 

    def callback(self, ros_data):
        '''Callback function of subscribed topic.  Here images get converted and features detected'''
        image_np = CvBridge().compressed_imgmsg_to_cv2(ros_data,desired_encoding='bgr8')
        if self.last is not None:
            print(np.sum(image_np-self.last))
        self.last = image_np
        
def main(args):
    '''Initializes and cleanup ros node'''
    ic = image_feature()
    rospy.init_node('image_feature', anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down ROS Image feature detector module")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
