"""
augment the junction with noisy intention 
used for AS6 junction
"""
import sys
sys.path.append('..')
from dataset import PioneerDataset as Dataset

COMPARED_PATH = 'test/compare.txt'
LABEL_PATH = 'test/label.txt'

frame = list()
intention_type = list()
current_velocity = list()
steering_wheel_angle = list()
dlm = list()

# read recorded data
with open(COMPARED_PATH,'r') as file:
    lines = file.readlines()[1:]
    for line in lines:
        tmp = line.split(" ")
        frame.append(tmp[0])
        intention_type.append(tmp[1])
        current_velocity.append(tmp[2])
        steering_wheel_angle.append(tmp[3])
        dlm.append(tmp[5][:-1]) #remove \n

# generate fake data 
with open(LABEL_PATH,'w') as file:
    file.write('frame intention_type current_velocity steering_wheel_angle dlm\n')
    for i in range(len(frame)):
        cur_frame = frame[i]
        cur_intention_type = intention_type[i]
        cur_current_velocity = current_velocity[i]
        cur_steering_wheel_angle = steering_wheel_angle[i]
        cur_dlm = dlm[i].lower()
        file.write(cur_frame+' '+cur_intention_type+' '+cur_current_velocity+' '+cur_steering_wheel_angle+' '+cur_dlm+'\n')

