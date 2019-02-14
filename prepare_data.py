import numpy as np
import pandas as pd

Train_Dir = './facial-keypoints-detection/training.csv'
Test_Dir = './facial-keypoints-detection/test.csv'
lookid_Dir = './input/IdLookupTable.csv'

# Read and save training data
data = pd.read_csv(Train_Dir)
num_images = data.shape[0]
images = np.zeros((num_images, 96, 96))
landmarks = np.zeros((num_images, data.shape[1] - 1))

for i in range(num_images):
    img = data['Image'][i].split(' ')
    img = np.array(img, dtype='float32').reshape(96, 96)
    images[i, :, :] = img

    ldmk = np.array(data.iloc[i, 0:-1], dtype='float32')
    landmarks[i, :] = ldmk

np.save('train_data.npy', images[0:images.shape[0]-1000])
np.save('train_ldmk.npy', landmarks[0:images.shape[0]-1000])
np.save('val_data.npy', images[images.shape[0]-1000:])
np.save('val_ldmk.npy', landmarks[images.shape[0]-1000:])

# Read and save test data

data = pd.read_csv(Test_Dir)
num_images = data.shape[0]
images = np.zeros((num_images, 96, 96))

for i in range(num_images):
    img = data['Image'][i].split(' ')
    img = np.array(img, dtype='float32').reshape(96, 96)
    images[i, :, :] = img

np.save('test_data.npy', images)
