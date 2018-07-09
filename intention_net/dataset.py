""" dataset for intention net"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import os.path as osp
import csv
import keras
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.resnet50 import preprocess_input
from keras.utils import to_categorical
from glob import glob

class BaseDataset(keras.utils.Sequence):
    NUM_CONTROL = 2
    def __init__(self, data_dir, batch_size, num_intentions, mode, target_size=(224, 224), shuffle=False, max_samples=None):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_intentions = num_intentions
        self.mode = mode
        self.target_size = target_size
        self.shuffle = shuffle
        self.max_samples = max_samples
        self.num_samples = None

        self.init()

        assert (self.num_samples is not None), "You must assign self.num_samples in init() function."

        if self.max_samples is not None:
            self.num_samples = min(self.max_samples, self.num_samples)

        self.on_epoch_end()

    def init(self):
        # you must assigh the num_samples here.
        raise NotImplementedError

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(self.num_samples)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __len__(self):
        """Denote number of batches per epoch"""
        return self.num_samples // self.batch_size

class CarlaSimDataset(BaseDataset):
    # intention mapping
    INTENTION_MAPPING = {}
    INTENTION_MAPPING[0] = 0
    INTENTION_MAPPING[2] = 1
    INTENTION_MAPPING[3] = 2
    INTENTION_MAPPING[4] = 3
    INTENTION_MAPPING[5] = 4

    NUM_CONTROL = 2
    def __init__(self, data_dir, batch_size, num_intentions, mode, target_size=(224, 224), shuffle=False, max_samples=None):
        super().__init__(data_dir, batch_size, num_intentions, mode, target_size, shuffle, max_samples)

    def init(self):
        self.labels = []
        self.files = []
        image_path_pattern = '_images/episode_{:s}/{:s}/image_{:0>5d}.jpg.png'
        frames = {}
        with open(osp.join(self.data_dir, 'measurements.csv'), 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                self.labels.append(row)
                episode_name = row['weather'] + '_' + row['exp_id'] + '_' + row['start_point'] + '.' + row['end_point']
                if episode_name in frames:
                    frames[episode_name] += 1
                else:
                    frames[episode_name] = 0
                fn = image_path_pattern.format(episode_name, 'CameraRGB', frames[episode_name])
                self.files.append(osp.join(self.data_dir, fn))
        self.num_samples = len(self.labels)

    def __getitem__(self, index):
        """Generate one batch of data"""
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        X = []
        I = []
        S = []
        Y = []
        for idx in indexes:
            lbl = self.labels[idx]
            img = load_img(self.files[idx], target_size=self.target_size)
            img = preprocess_input(img_to_array(img))
            intention = to_categorical(self.INTENTION_MAPPING[int(float(lbl['intention']))], num_classes=self.num_intentions)
            speed = [float(lbl['speed'])]
            control = [float(lbl['steer']), float(lbl['throttle'])-float(lbl['brake'])]
            X.append(img)
            I.append(intention)
            S.append(speed)
            Y.append(control)
        X = np.array(X)
        I = np.array(I)
        S = np.array(S)
        Y = np.array(Y)
        return [X, I, S], Y

class CarlaImageDataset(CarlaSimDataset):
    STEER = 0
    GAS = 1
    BRAKE = 2
    HAND_BRAKE = 3
    REVERSE_GEAR = 4
    STEER_NOISE = 5
    GAS_NOISE = 6
    BRAKE_NOISE = 7
    POS_X = 8
    POS_Y = 9
    SPEED = 10
    COLLISION_OTHER = 11
    COLLISION_PEDESTRIAN = 12
    COLLISION_CAR = 13
    OPPOSITE_LANE_INTER = 14
    SIDEWALK_INTER = 15
    ACC_X = 16
    ACC_Y = 17
    ACC_Z = 18
    PLATFORM_TIME = 19
    GAME_TIME = 20
    ORIENT_X = 21
    ORIENT_Y = 22
    ORIENT_Z = 23
    INTENTION = 24
    NOISE = 25
    CAMERA = 26
    CAMERA_YAW = 27

    def __init__(self, data_dir, batch_size, num_intentions, mode, target_size=(224, 224), shuffle=False, max_samples=None):
        super().__init__(data_dir, batch_size, num_intentions, mode, target_size, shuffle, max_samples)

    def init(self):
        self.labels = np.loadtxt(osp.join(self.data_dir, 'label.txt'))
        self.num_samples = self.labels.shape[0]

        self.files = [self.data_dir + '/' + str(fn)+'.png' for fn in self.labels[:,0].astype(np.int32)][:self.num_samples]
        if self.mode.startswith('LPE'):
            self.lpe_files = [self.data_dir + '/lpe_' + str(fn)+'.png' for fn in self.labels[:,0].astype(np.int32)][:self.num_samples]

        self.labels = self.labels[:,1:]

    def __getitem__(self, index):
        """Generate one batch of data"""
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        X = []
        I = []
        S = []
        Y = []
        for idx in indexes:
            lbl = self.labels[idx]
            img = load_img(self.files[idx], target_size=self.target_size)
            img = preprocess_input(img_to_array(img))
            if self.mode == 'DLM':
                intention = to_categorical(self.INTENTION_MAPPING[lbl[self.INTENTION]], num_classes=self.num_intentions)
            else:
                intention = load_img(self.lpe_files[idx], target_size=self.target_size)
                intention = preprocess_input(img_to_array(intention))
            # transfer from km/h to m/s
            speed = [lbl[self.SPEED]/3.6]
            control = [lbl[self.STEER], lbl[self.GAS]-lbl[self.BRAKE]]
            X.append(img)
            I.append(intention)
            S.append(speed)
            Y.append(control)
        X = np.array(X)
        I = np.array(I)
        S = np.array(S)
        Y = np.array(Y)
        return [X, I, S], Y

class HuaWeiDataset(BaseDataset):
    def __init__(self, data_dir, batch_size, num_intentions, mode, target_size=(224, 224), shuffle=False, max_samples=None):
        super().__init__(data_dir, batch_size, num_intentions, mode, target_size, shuffle, max_samples)

    def init(self):
        routes = glob(os.path.join(self.data_dir, 'Log*'))
        self.labels = []
        self.num_samples = 0
        for route in routes:
            self.car_data_header, self.car_data = self.read_csv(os.path.join(route, 'LabelData_VehicleData_PRT.txt'))
            self.labels.append(self.car_data)
            self.num_samples += len(self.car_data)
        # reverse header index
        self.car_data_idx = {}
        for k, v in enumerate(self.car_data_header):
            self.car_data_idx[v] = k
        self.global_map = None

    def get_global_map(self, lat_min, lon_min, lat_max, lon_max, z=18):
        import smopy
        print (lat_min, lon_min, lat_max, lon_max)
        map = smopy.Map((lat_min, lon_min, lat_max, lon_max), z=z)
        self.global_map = map

    # for debug use
    def plot_trajectory(self):
        import matplotlib.pyplot as plt
        #convert from gcj2 to wgs-84
        from coord_convert.utils import Transform
        transform = Transform()
        longitudes = []
        latitudes = []
        for label in self.labels:
            for data in label:
                lon = float(data[self.car_data_idx['longitude']])
                lat = float(data[self.car_data_idx['latitude']])
                #wgs_lon, wgs_lat = transform.gcj2wgs(lon, lat)
                wgs_lon, wgs_lat = lon, lat
                longitudes.append(wgs_lon)
                latitudes.append(wgs_lat)
        self.get_global_map(min(latitudes), min(longitudes), max(latitudes), max(longitudes))
        pixels = [self.global_map.to_pixels(lat, lon) for lat, lon in zip(latitudes, longitudes)]
        x, y = zip(*pixels)
        ax = self.global_map.show_mpl(figsize=(8, 6))
        ax.plot(x, y, 'ro')
        plt.show()

    def read_csv(self, fn, has_header=True):
        f = open(fn)
        reader = csv.reader(f, delimiter=' ')
        header = None
        data = []
        if has_header:
            row_num = 0
            for row in reader:
                if row_num == 0:
                    header = row
                    row_num += 1
                else:
                    data.append(row)
                    row_num += 1
        else:
            for row in reader:
                data.append(row)

        return header, data

    def __getitem__(self, index):
        pass

intention_mapping = CarlaSimDataset.INTENTION_MAPPING

def test():
    #d = CarlaSimDataset('/home/gaowei/SegIRLNavNet/_benchmarks_results/Debug', 2, 5, max_samples=10)
    #d = CarlaImageDataset('/media/gaowei/Blade/linux_data/carla_data/AgentHuman/ImageData', 2, 5, mode='LPE_SIAMESE', max_samples=10)
    d = HuaWeiDataset('/media/gaowei/Blade/linux_data/HuaWeiData', 2, 5, 'DLM', max_samples=10)
    d.plot_trajectory()
    """
    for step, (x,y) in enumerate(d):
        print (x[0].shape, x[1].shape, x[2].shape, y.shape)
        print (x[2])
        if step == len(d)-1:
            break
    """

#test()
