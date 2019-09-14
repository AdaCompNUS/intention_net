from torch.utils.data import Dataset
import os.path as osp
import os 
from PIL import Image
from skimage import io, transform
from skimage.color import rgb2gray

def read_csv(path):
    data = dict()
    label = dict()
    with open(path,'r') as file:
        lines = file.readlines()[1:]
        for line in lines:
            line = line[:-1]
            tmp = line.split(" ")
            data[tmp[0]] = tmp[4]#remove \n
            label[tmp[0]] = [tmp[2],tmp[3]]
    return data,label

class PioneerDataset(Dataset):
    SCALE_VEL = 0.5
    SCALE_STEER = 0.5
    INTENTION_MAPPING = {
        'left' : 1,
        'right': 2,
        'forward':0,
        'stop':3,
    }

    def __init__(self,data_dir,target_size=(224,224),transform=None,BASE_DIR='data',LEFT_DIR='rgb_1',RIGHT_DIR='rgb_2',FRONT_DIR='rgb_0'):
        super(PioneerDataset,self).__init__()
        self.data_dir = data_dir
        self.target_size = target_size
        self.transform = transform
        self.BASE_DIR = BASE_DIR
        self.LEFT_DIR = LEFT_DIR
        self.FRONT_DIR = FRONT_DIR
        self.RIGHT_DIR = RIGHT_DIR

        base_dir = osp.join(self.data_dir, self.BASE_DIR)
        self.data,self.label = read_csv(os.path.join(base_dir, 'label.txt')) # data: map img_fn -> intention, label: map img_fn->(vel,turning_angle)
        self.index = list(self.data.keys()) # list of img
        self.num_samples = len(self.data)

    def __len__(self):
        return self.num_samples

    def __getitem__(self,idx):
        # get file name 
        fn = self.index[idx]
        intention = self.INTENTION_MAPPING[self.data[fn]]

        # read img
        lbnw_im_path = osp.join(self.data_dir,self.BASE_DIR,self.LEFT_DIR,fn+".jpg")
        mbnw_im_path = osp.join(self.data_dir,self.BASE_DIR,self.FRONT_DIR,fn+".jpg")
        rbnw_im_path = osp.join(self.data_dir,self.BASE_DIR,self.RIGHT_DIR,fn+".jpg")
        dl_im_path = osp.join(self.data_dir,self.BASE_DIR,self.LEFT_DIR,fn+".jpg")
        dm_im_path = osp.join(self.data_dir,self.BASE_DIR,self.FRONT_DIR,fn+".jpg")
        dr_im_path = osp.join(self.data_dir,self.BASE_DIR,self.RIGHT_DIR,fn+".jpg")

        lbnw = io.imread(lbnw_im_path)
        mrgb = io.imread(mbnw_im_path)
        mbnw = rgb2gray(mrgb)
        rbnw = io.imread(rbnw_im_path)
        dl = io.imread(dl_im_path)
        dm = io.imread(dm_im_path)
        dr = io.imread(dr_im_path)

        if self.transform:
            lbnw = self.transform(lbnw)
            mrgb = self.transform(mbnw)
            rbnw = self.transform(rbnw)
            dl = self.transform(dl)
            dm = self.transform(dm)
            dr = self.transform(dr)
        
        target = self.label[fn]
        
        return [intention,lbnw,mbnw,rbnw,dl,dm,dr],target
