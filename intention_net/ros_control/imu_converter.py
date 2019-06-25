import rospy

from sensor_msgs.msg import Imu

def cb_imu(msg):
	gyro = msg.angular_velocity
	acc = msg.linear_acceleration

	# Create new Imu Message

	imu = Imu()
	imu.header.frame_id = "mynteye_link"
	imu.header.stamp = msg.header.stamp
	imu.orientation = msg.orientation

	imu.angular_velocity.x = gyro.z
	imu.angular_velocity.y = -gyro.y
	imu.angular_velocity.z = gyro.x

	imu.linear_acceleration.x = acc.z
	imu.linear_acceleration.y = -acc.y
	imu.linear_acceleration.z = acc.x

	imu_pub.publish(imu)


rospy.init_node("imu_transform")

rospy.Subscriber('/mynteye/imu/data_raw', Imu, cb_imu, queue_size=1, buff_size=2**10)
imu_pub = rospy.Publisher('/imu', Imu, queue_size=1)

rospy.spin()


