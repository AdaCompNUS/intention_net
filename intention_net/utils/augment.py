"""
augment the junction with noisy intention
"""
frame = list()
intention_type = list()
current_velocity = list()
steering_wheel_angle = list()
dlm = list()

INTENTION_MAPPING = {'forward':0,
                    'left':1,
                    'right':2,
                    'stop':0}
REVERSE_INTENTION = { 0: 'forward',
                    1 : 'left',
                    2 : 'right'}

def merry_go_round(intention):
    intention_int = INTENTION_MAPPING[intention]
    return REVERSE_INTENTION[((intention_int+1)%3)]

# read recorded data
with open('label.txt','r') as file:
    lines = file.readlines()[1:]
    for line in lines:
        tmp = line.split(" ")
        frame.append(tmp[0])
        intention_type.append(tmp[1])
        current_velocity.append(tmp[2])
        steering_wheel_angle.append(tmp[3])
        dlm.append(tmp[4])

# generate fake data 
with open('fake.txt','w') as file:
    file.write('frame intention_type current_velocity steering_wheel_angle dlm\n')
    for _ in range(len(REVERSE_INTENTION)):
        for i in range(len(frame)):
            cur_frame = frame[i]
            cur_intention_type = intention_type[i]
            cur_current_velocity = current_velocity[i]
            cur_steering_wheel_angle = steering_wheel_angle[i]
            cur_dlm = merry_go_round(dlm[i])
            file.write(cur_frame+' '+cur_intention_type+' '+cur_current_velocity+' '+cur_steering_wheel_angle+' '+cur_dlm+'\n')

