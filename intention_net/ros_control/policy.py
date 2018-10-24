"""
load the learned model
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import numpy as np
import scipy.misc

from keras.utils import to_categorical
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

# intention net package
from intention_net.net import IntentionNet
from intention_net.dataset import preprocess_input

class Policy(object):
    def __init__(self, mode, input_frame, num_control, path, num_intentions, gpu_fraction=0.75):
        # set keras session
        config_gpu = tf.ConfigProto()
        config_gpu.gpu_options.allow_growth = True
        config_gpu.gpu_options.per_process_gpu_memory_fraction=gpu_fraction
        KTF.set_session(tf.Session(config=config_gpu))
        self.model = None
        self.mode = mode
        self.input_frame = input_frame
        self.num_control = num_control
        self.num_intentions = num_intentions
        self.path = path
        self.load_model()

    def load_model(self):
        model = IntentionNet(self.mode, self.input_frame, self.num_control, self.num_intentions)
        # load checkpoint
        #fn = osp.join(self.path, self.input_frame + '_' + self.mode+'_best_model.h5')
        fn = osp.join(self.path, self.input_frame + '_' + self.mode+'_latest_model.h5')
        model.load_weights(fn)
        print ("=> loaded checkpoint '{}'".format(fn))
        self.model = model

    def predict_control(self, image, intention, speed=None):
        rgb = scipy.misc.imresize(image, (224, 224))
        rgb = np.expand_dims(preprocess_input(rgb), axis=0)

        if self.mode == 'DLM':
            intention = to_categorical([intention], num_classes=self.num_intentions)
        else:
            intention = np.expand_dims(preprocess_input(intention), axis=0)

        if speed is not None:
            speed = np.array([[speed]])

            pred_control = self.model.predict([rgb, intention, speed])
            return pred_control
        else:
            pred_control = self.model.predict([rgb, intention])
            return pred_control
