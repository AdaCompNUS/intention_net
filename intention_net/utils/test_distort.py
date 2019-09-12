import rospy
from sensor_msgs.msg import Image 
from cv_bridge import CvBridge
from undistort import undistort
import numpy as np

def cb(msg):
    img = CvBridge().imgmsg_to_cv2(msg,desired_encoding='bgr8')
    img = undistort(img)
    msg = CvBridge().cv2_to_imgmsg(img)
    pub.publish(msg)

if __name__ == '__main__':
    rospy.init_node('undistort')
    sub = rospy.Subscriber('/mynteye/left/image_raw',Image, cb,queue_size=1,buff_size=2*10)
    pub = rospy.Publisher('/undistort',Image,queue_size=1)
    rospy.spin()