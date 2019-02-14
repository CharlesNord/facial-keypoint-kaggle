from load_data import get_train_loader, get_val_loader, get_test_loader
import matplotlib.pyplot as plt
from utils import plot_grid, get_grid
from models import Model
from torch import optim
from tensorboardX import SummaryWriter
from PIL import Image
import torch

# Get train loader and val loader and test loader
train_loader = get_train_loader()
val_loader = get_val_loader()
test_loader = get_test_loader()

# Test train loader
sample = iter(val_loader).next()
imgs, lms = sample['image'], sample['landmarks']
plot_grid(imgs, lms)


# Loss function
def loss_func(pred, gt):
    batch_sz = pred.shape[0]
    diff = pred - gt.view(batch_sz, -1)
    nan_ind = (diff!=diff)
    diff[nan_ind] = 0
    loss = diff.pow(2).sum() / (diff.numel()-nan_ind.sum())
    return loss


# initialize the model
model = Model()
model.cuda()

# initialize the optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# tensorboard
writer = SummaryWriter('log')
total_train_step = 0
total_val_step = 0


def train(epoch):
    global total_train_step
    model.train()
    train_loss = 0
    for batch_idx, samples in enumerate(train_loader):

        total_train_step += 1

        imgs, lms = samples['image'], samples['landmarks']
        imgs = imgs.cuda()
        lms = lms.cuda()

        optimizer.zero_grad()
        pred = model(imgs)
        loss = loss_func(pred, lms)
        loss.backward()
        train_loss += loss.item()
        writer.add_scalar('train loss', loss.item(), total_train_step)
        optimizer.step()
        if batch_idx % 50 == 0:
            print('Train epoch: {}\t Step: {}\t Loss: {}'.format(epoch, batch_idx, loss.item()))

    train_loss /= len(train_loader)
    writer.add_scalar('avg/train', train_loss, epoch)
    print('======> Epoch: {}\t Average Train Loss: {}'.format(epoch, train_loss))
    return train_loss


def eval(epoch):
    model.eval()
    global total_val_step
    test_loss = 0
    for batch_idx, samples in enumerate(val_loader):
        total_val_step += 1
        imgs, lms = samples['image'], samples['landmarks']
        imgs = imgs.cuda()
        lms = lms.cuda()

        pred = model(imgs)
        loss = loss_func(pred, lms)
        test_loss += loss.item()
        writer.add_scalar('val loss', loss.item(), total_val_step)

    test_loss /= len(val_loader)
    writer.add_scalar('avg/val', test_loss, epoch)
    print('=====> Epoch: {}\t Average Test Loss: {}'.format(epoch, test_loss))
    return test_loss


def test(epoch):
    imgs = iter(test_loader).next()
    imgs = imgs.cuda()
    pred = model(imgs)
    I = get_grid(imgs.detach().cpu(), pred.detach().cpu())
    I = Image.fromarray(I.numpy())
    I.save('./results/{}.jpg'.format(epoch))


for epoch in range(150):
    train_loss = train(epoch)
    test_loss = eval(epoch)
    test(epoch)

stat_dict = model.state_dict()
torch.save(stat_dict,'model.pkl')