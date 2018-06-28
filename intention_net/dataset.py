""" dataset for intention net"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os.path as osp
import csv
import keras
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.resnet50 import preprocess_input
from keras.utils import to_categorical


class CarlaSimDataset(keras.utils.Sequence):
    # intention mapping
    INTENTION_MAPPING = {}
    INTENTION_MAPPING[0] = 0
    INTENTION_MAPPING[2] = 1
    INTENTION_MAPPING[3] = 2
    INTENTION_MAPPING[4] = 3
    INTENTION_MAPPING[5] = 4

    NUM_CONTROL = 2
    def __init__(self, data_dir, batch_size, num_intentions, target_size=(224, 224), shuffle=False, max_samples=None):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_intentions = num_intentions
        self.shuffle = shuffle
        self.target_size = target_size

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
        if max_samples is not None:
            self.num_samples = min(max_samples, self.num_samples)

        self.on_epoch_end()

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

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(self.num_samples)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __len__(self):
        """Denote number of batches per epoch"""
        return self.num_samples // self.batch_size

intention_mapping = CarlaSimDataset.INTENTION_MAPPING

def test():
    d = CarlaSimDataset('/home/gaowei/SegIRLNavNet/_benchmarks_results/Debug', 2, 5, max_samples=10)
    for step, (x,y) in enumerate(d):
        print (x[0].shape, x[1].shape, x[2].shape, y.shape)
        if step == len(d)-1:
            break

#test()
