import random
from argparse import ArgumentParser
import sys
sys.path.append('..')
sys.path.append('/mnt/intention_net')

import torch
from torch.optim import Adam,SGD,RMSprop
from torch.utils.data import DataLoader
from torch import nn
from torchvision import transforms
from torch.nn import functional as F

from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events, create_supervised_evaluator, create_supervised_trainer
from ignite.handlers import ModelCheckpoint,Timer
from ignite.metrics import Loss

from tensorboardX import SummaryWriter

from src.model import DepthIntentionEncodeModel
from src.dataset import MultiCamPioneerDataset

def check_manual_seed(seed):
    """ If manual seed is not specified, choose a random one and communicate it to the user.
    """

    seed = seed or random.randint(1, 10000)
    random.seed(seed)
    torch.manual_seed(seed)

    print('Using manual seed: {seed}'.format(seed=seed))
    
def get_dataloader(train_dir,val_dir=None,use_transform=False,num_workers=1,batch_size=16,shuffle=False):
    data_transform = transforms.Compose([transforms.ToTensor()])
    if use_transform:
        train_data = MultiCamPioneerDataset(train_dir,transform=data_transform)
    else:
        train_data = MultiCamPioneerDataset(train_dir,transform=None)

    train_loader = DataLoader(train_data,batch_size=batch_size,shuffle=shuffle,num_workers=num_workers)
    
    if val_dir: 
        val_data = MultiCamPioneerDataset(val_dir,data_transform)
        val_loader = DataLoader(val_data,batch_size=batch_size,shuffle=shuffle,num_workers=num_workers)
    else:
        val_loader = None 
    
    return train_loader,val_loader

def create_summary_writer(model,data_loader,log_dir):
    writer = SummaryWriter(log_dir=log_dir)
    data_loader_iter = iter(data_loader)
    x,y = next(data_loader_iter)
    try:
        writer.add_graph(model,x)
    except Exception as e:
        print('Failed to save graph: {}'.format(e))
    return writer

def run(train_dir,val_dir=None,learning_rate=1e-4,num_workers=1,num_epochs=100,batch_size=16,shuffle=False,num_controls=2,num_intentions=4,hidden_dim=256,log_interval=10,log_dir='./logs',seed=2605,accumulation_steps=4):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    train_loader,val_loader = get_dataloader(train_dir,val_dir,num_workers=num_workers,batch_size=batch_size,shuffle=shuffle)
    model = DepthIntentionEncodeModel(num_controls=num_controls,num_intentions=num_intentions,hidden_dim=hidden_dim)
    model.to(device)
    writer = create_summary_writer(model,train_loader,log_dir)
    criterion = nn.MSELoss()
    check_manual_seed(seed)
    #TODO: change to RAdam
    optim = Adam(model.parameters(),lr=learning_rate)

    def update_fn(engine, batch):
        model.train()
        if engine.state.iteration-1 % accumulation_steps == 0:
            #engine.state.cummulative_loss = 0.0
            optim.zero_grad()

        x, y = batch
        for elem in x:
            elem = elem.to(device)
        for elem in y:
            elem = elem.to(device)

        y_pred = model(*x)
        # if engine.state.iteration % 16:
        #     print(y)
        #     print(y_pred)
        #     print(x[0])
        loss = criterion(y_pred, y) / accumulation_steps
        loss.backward()

        #engine.state.cummulative_loss += loss

        if engine.state.iteration-1 % accumulation_steps == 0:
            optim.step()

        return loss.item()
    
    def evaluate_fn(engine,batch):
        engine.state.metrics = dict()
        model.eval()

        x,y = batch
        
        x.to(device)
        y.to(device)

        y_pred = model(*x)
        mse_loss = F.mse_loss(y_pred,y)
        mae_loss = F.l1_loss(y_pred,y)

        engine.state.metrics['mse'] = mse_loss
        engine.state.metrics['mae'] = mae_loss

    trainer = Engine(update_fn)
    evaluator = Engine(evaluate_fn)
    
    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        iter = (engine.state.iteration - 1)%len(train_loader)+1
        if iter % log_interval == 0:
            print("[Epoch: {}][Iteration: {}/{}] loss: {:.4f}".format(engine.state.epoch,iter,len(train_loader),engine.state.output))
            writer.add_scalar("training/loss",engine.state.output,engine.state.iteration)
    
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        evaluator.run(train_loader)
        metrics = evaluator.state.metrics
        mse = metrics['mse']
        mae = metrics['mae']
        print("Training Results - Epoch: {}  mae: {:.2f} mse: {:.2f}".format(engine.state.epoch, mse, mae))
        writer.add_scalar("training/mse", mse, engine.state.epoch)
        writer.add_scalar("training/mae", mae, engine.state.epoch)
    
    trainer.run(train_loader,max_epochs=num_epochs)
    writer.close()



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16,
                        help='input batch size for training (default: 16)')
    parser.add_argument('--val_batch_size', type=int, default=16,
                        help='input batch size for validation (default: 1000)')
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--log_interval', type=int, default=2,
                        help='how many batches to wait before logging training status')
    parser.add_argument("--log_dir", type=str, default="logs",
                        help="log directory for Tensorboard log output")
    parser.add_argument('--train_dir',type=str,default="/Users/lhduong/Downloads/sample",help="path to train data directory")
    parser.add_argument('--val_dir',type=str,default="/Users/lhduong/Downloads/val",help="path to val data directory")
    parser.add_argument('--shuffle',type=bool,default=False,help="Choose to shuffle the training set")
    parser.add_argument('--num_intentions',type=int,default=4,help="number of intentions")
    parser.add_argument('--num_controls',type=int,default=2,help="number of controls")
    parser.add_argument('--hidden_dim',type=int,default=256,help="hidden size of image embedded")
    parser.add_argument('--accumulation_steps',type=int,default=1,help="number of accumulation steps for gradient update")
    args = parser.parse_args()

    run(train_dir=args.train_dir,val_dir=None,learning_rate=args.lr,num_workers=1,
        num_epochs=args.num_epochs,batch_size=args.batch_size,
        shuffle=args.shuffle,num_controls=args.num_controls,num_intentions=args.num_intentions,
        hidden_dim=args.hidden_dim,log_interval=args.log_interval,log_dir=args.log_dir,seed=2605,
        accumulation_steps=args.accumulation_steps)

# def update_fn(engine, batch):
#     model.train()

#     if engine.state.iteration % accumulation_steps == 0:
#         optimizer.zero_grad()

#     x, y = prepare_batch(batch, device=device, non_blocking=non_blocking)
#     y_pred = model(x)
#     loss = criterion(y_pred, y) / accumulation_steps
#     loss.backward()

#     if engine.state.iteration % accumulation_steps == 0:
#         optimizer.step()

#     return loss.item()

# trainer = Engine(update_fn)

# def train(seed,device):
    # check_manual_seed(seed)