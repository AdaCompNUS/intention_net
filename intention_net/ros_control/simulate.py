"""
simulate rostopics from dataset for testing
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import fire
import numpy as np
from tqdm import tqdm

# ros packages
import rospy
from std_msgs.msg import Float32, Int32
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
import cv2
from cv_bridge import CvBridge

def main_wrapper(dataset, data_dir, num_intentions=5, mode='DLM'):
    if dataset == 'CARLA':
        from intention_net.dataset import CarlaImageDataset as Dataset
        print ('=> use CARLA published data')
    elif dataset == 'CARLA_SIM':
        from intention_net.dataset import CarlaSimDataset as Dataset
        print ('=> use CARLA self-collected data')
    else:
        from intention_net.dataset import HuaWeiFinalDataset as Dataset
        print ('=> use HUAWEI data')

    # create rosnode
    rospy.init_node('simulate')
    # only for debug so make it slow
    rate = rospy.Rate(5)
    rgb_pub = rospy.Publisher('/image', Image, queue_size=1)
    speed_pub = rospy.Publisher('/speed', Float32, queue_size=1)
    control_pub = rospy.Publisher('/labeled_control', Twist, queue_size=1)
    if mode == 'DLM':
        intention_pub = rospy.Publisher('/intention', Int32, queue_size=1)
    else:
        intention_pub = rospy.Publisher('/intention', Image, queue_size=1)

    sim_loader = Dataset(data_dir, 1, num_intentions, mode, preprocess=False)
    for step, (x, y) in enumerate(tqdm(sim_loader)):
        img = x[0][0].astype(np.uint8)
        img = CvBridge().cv2_to_imgmsg(img, encoding='bgr8')
        intention = x[1][0]
        if mode == 'DLM':
            intention = np.argmax(intention)
        else:
            intention = intention.astype(np.uint8)
            intention = CvBridge().cv2_to_imgmsg(intention, encoding='bgr8')
        speed = x[2][0, 0]
        control = y[0]
        twist = Twist()
        twist.linear.x = control[1]
        twist.angular.z = control[0]
        # publish topics
        rgb_pub.publish(img)
        intention_pub.publish(intention)
        speed_pub.publish(speed)
        control_pub.publish(twist)
        rate.sleep()

def main():
    fire.Fire(main_wrapper)
    
if __name__ == '__main__':
    main()
