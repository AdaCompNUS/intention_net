from dataset import PioneerDataset
from model import DepthIntentionEncodeModel
from torch.optim import Adam,SGD
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def train():
    