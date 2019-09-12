"""
use sliding window to interpolate the intention
"""
import sys
sys.path.append('..')
from dataset import PioneerDataset as Dataset
from collections import deque
from functools import reduce
import numpy as np
import math

# forward_weight, left_weight, right_weight, stop_weight
# DEPENDS ON Dataset.INTENTION_MAPPING ORDER
WEIGHT = np.array([1,3,3,4]).reshape(1,-1)
WINDOW_SIZE = 15
LABEL_PATH = 'test/label.txt'
COMPARE_PATH = 'test/compare.txt'
INTERPOLATE_PATH = 'test/interpolated.txt'

def window(seq, n=WINDOW_SIZE):
    if len(seq) < n:
        print('size of queue is bigger than sequence length')
        return seq
    else:
        q = deque(maxlen=n)
        ret = list()

        for i in range(n):
            q.append(seq[i])
        ret.append(list(q))
        
        for i in range(n,len(seq)):
            q.append(seq[i])
            ret.append(list(q))
        return ret

def id2onehot(indices,n_classes=4):
    indices = np.array(indices).reshape(-1)
    one_hot = np.eye(n_classes)[indices]
    return one_hot

def denoise_intention(intentions):
    mapped_intention = np.array(list(map(lambda x: Dataset.INTENTION_MAPPING[x],intentions))).reshape(-1)
    one_hot = id2onehot(mapped_intention,len(Dataset.INTENTION_MAPPING))
    weighted = one_hot*WEIGHT
    voted_value = np.sum(weighted,axis=0)
    intention = np.argmax(voted_value)
    return Dataset.INTENTION_MAPPING_NAME[intention]
    
def main():
    # Initialize
    frame = list()
    intention_type = list()
    current_velocity = list()
    steering_wheel_angle = list()
    dlm = list()

    with open(LABEL_PATH,'r') as file:
        lines = file.readlines()[1:]
        for line in lines:
            tmp = line.split(" ")
            frame.append(tmp[0])
            intention_type.append(tmp[1])
            current_velocity.append(tmp[2])
            steering_wheel_angle.append(tmp[3])
            dlm.append(tmp[4][:-1]) #remove \n

    sliding_window = window(dlm)
    prev_dlm = dlm
    dlm = list(map(denoise_intention,sliding_window))

    with open(INTERPOLATE_PATH,'w') as f1:
        f1.write('frame intention_type current_velocity steering_wheel_angle dlm\n')
        for i in range(len(dlm)):
            cur_frame = frame[i+math.floor(WINDOW_SIZE/2)]
            cur_intention_type = intention_type[i+math.floor(WINDOW_SIZE/2)]
            cur_current_velocity = current_velocity[i+math.floor(WINDOW_SIZE/2)]
            cur_steering_wheel_angle = steering_wheel_angle[i+math.floor(WINDOW_SIZE/2)]
            cur_dlm = dlm[i]
            f1.write(cur_frame+' '+cur_intention_type+' '+cur_current_velocity+' '+cur_steering_wheel_angle+' '+cur_dlm+'\n')
        f1.close()

    # write another file to compare
    with open(COMPARE_PATH,'w') as f2:
        f2.write('frame intention_type current_velocity steering_wheel_angle dlm prev_dlm\n')
        for i in range(len(dlm)):
            # SHIFT OTHER DOWN
            cur_frame = frame[i+math.floor(WINDOW_SIZE/2)]
            cur_intention_type = intention_type[i+math.floor(WINDOW_SIZE/2)]
            cur_current_velocity = current_velocity[i+math.floor(WINDOW_SIZE/2)]
            cur_steering_wheel_angle = steering_wheel_angle[i+math.floor(WINDOW_SIZE/2)]
            cur_dlm = dlm[i]
            last_dlm = prev_dlm[i+math.floor(WINDOW_SIZE/2)]
            if last_dlm == cur_dlm:
                f2.write(cur_frame+' '+cur_intention_type+' '+cur_current_velocity+' '+cur_steering_wheel_angle+' '+cur_dlm+' '+last_dlm+'\n')
            else:
                f2.write(cur_frame+' '+cur_intention_type+' '+cur_current_velocity+' '+cur_steering_wheel_angle+' '+cur_dlm+' '+last_dlm.upper()+'\n')
        f2.close()
        
if __name__ == '__main__':
    main()