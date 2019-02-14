from load_data import get_train_loader, get_val_loader, get_test_loader
import matplotlib.pyplot as plt
from utils import plot_grid, get_grid
from models import Model
from torch import optim
from tensorboardX import SummaryWriter
from PIL import Image
import torch

import numpy as np
import pandas as pd
#
# Train_Dir = './facial-keypoints-detection/training.csv'
# Test_Dir = './facial-keypoints-detection/test.csv'
# lookid_Dir = './input/IdLookupTable.csv'
#
# # Read and save training data
# data = pd.read_csv(Train_Dir)
# num_images = data.shape[0]
# images = np.zeros((num_images, 96, 96))
# landmarks = np.zeros((num_images, data.shape[1] - 1))
#
# for i in range(num_images):
#     img = data['Image'][i].split(' ')
#     img = np.array(img, dtype='float32').reshape(96, 96)
#     images[i, :, :] = img
#
#     ldmk = np.array(data.iloc[i, 0:-1], dtype='float32')
#     landmarks[i, :] = ldmk
#
# np.save('train_data_total.npy', images)
# np.save('train_ldmk_total.npy', landmarks)
#
# # Get train loader and val loader and test loader
# train_loader = get_train_loader()
# test_loader = get_test_loader()
#
#
# # Loss function
# def loss_func(pred, gt):
#     batch_sz = pred.shape[0]
#     diff = pred - gt.view(batch_sz, -1)
#     nan_ind = (diff != diff)
#     diff[nan_ind] = 0
#     loss = diff.pow(2).sum() / (diff.numel() - nan_ind.sum())
#     return loss
#
#
# # initialize the model
# model = Model()
# model.cuda()
#
# # initialize the optimizer
# optimizer = optim.Adam(model.parameters(), lr=1e-4)
#
# # tensorboard
# writer = SummaryWriter('log')
# total_train_step = 0
# total_val_step = 0
#
#
# def train(epoch):
#     global total_train_step
#     model.train()
#     train_loss = 0
#     for batch_idx, samples in enumerate(train_loader):
#
#         total_train_step += 1
#
#         imgs, lms = samples['image'], samples['landmarks']
#         imgs = imgs.cuda()
#         lms = lms.cuda()
#
#         optimizer.zero_grad()
#         pred = model(imgs)
#         loss = loss_func(pred, lms)
#         loss.backward()
#         train_loss += loss.item()
#         writer.add_scalar('train loss', loss.item(), total_train_step)
#         optimizer.step()
#         if batch_idx % 50 == 0:
#             print('Train epoch: {}\t Step: {}\t Loss: {}'.format(epoch, batch_idx, loss.item()))
#
#     train_loss /= len(train_loader)
#     writer.add_scalar('avg/train', train_loss, epoch)
#     print('======> Epoch: {}\t Average Train Loss: {}'.format(epoch, train_loss))
#     return train_loss
#
#
# def test(epoch):
#     imgs = iter(test_loader).next()
#     imgs = imgs.cuda()
#     pred = model(imgs)
#     I = get_grid(imgs.detach().cpu(), pred.detach().cpu())
#     I = Image.fromarray(I.numpy())
#     I.save('./results/{}.jpg'.format(epoch))
#
#
# for epoch in range(300):
#     train_loss = train(epoch)
#     test(epoch)
#
# stat_dict = model.state_dict()
# torch.save(stat_dict, 'model.pkl')

model = Model()
model.cuda()
stat_dict = torch.load('model.pkl')
model.load_state_dict(stat_dict)
test_loader = get_test_loader()
test_data = test_loader.dataset
pred = []
for idx in range(len(test_data)):
    img = test_data[idx].unsqueeze(0)
    pt = model(img.cuda())
    pred.append(pt)
pred = torch.stack(pred, dim=0).detach().squeeze().cpu().numpy()

lookid_data = pd.read_csv('facial-keypoints-detection/IdLookupTable.csv')
lookid_list = list(lookid_data['FeatureName'])
imageID = list(lookid_data['ImageId'] - 1)
rowid = lookid_data['RowId']
rowid = list(rowid)
feature = []
for f in list(lookid_list):
    feature.append(lookid_list.index(f))

pre_list = list(pred)
preded = []
for x, y in zip(imageID, feature):
    preded.append(pre_list[x][y])

rowid = pd.Series(rowid, name='RowId')
loc = pd.Series(preded, name='Location')
submission = pd.concat([rowid, loc], axis=1)
submission.to_csv('face_key_detection_submission.csv',index=False)
