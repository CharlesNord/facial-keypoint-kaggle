import numpy as np
from torch.utils.data import Dataset
import torch.utils.data as Data
import matplotlib.pyplot as plt
from torchvision import transforms
from skimage import transform
import torch
import cv2


class Rescale(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)
        img = transform.resize(image, (new_h, new_w))
        landmarks = landmarks * [new_w * 1.0 / w, new_h * 1.0 / h]

        return {'image': img, 'landmarks': landmarks}


class RandomCrop(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        h, w = image.shape[:2]
        new_h, new_w = self.output_size
        if h > new_h:
            top = np.random.randint(0, h - new_h)
        else:
            top = 0
        if w > new_w:
            left = np.random.randint(0, w - new_w)
        else:
            left = 0
        image = image[top:top + new_h,
                left:left + new_w]
        landmarks = landmarks - [left, top]
        return {'image': image, 'landmarks': landmarks}


class RandomFlip(object):
    def __init__(self, p=0.5):
        self.p = 0.5

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        p = np.random.rand()
        if p > self.p:
            image = np.fliplr(image)
            landmarks = landmarks * [-1, 1] + [96, 0]

        return {'image': image, 'landmarks': landmarks}


def get_rotation_mat(img, angle_degree):
    height, width = img.shape[0:2]
    center = (width / 2.0, height / 2.0)
    angle_radians = np.radians(angle_degree)
    rot_mat = cv2.getRotationMatrix2D(center, angle_degree, scale=1)
    new_height = height * np.abs(np.cos(angle_radians)) + width * np.abs(np.sin(angle_radians))
    new_width = height * np.abs(np.sin(angle_radians)) + width * np.abs(np.cos(angle_radians))
    new_center = (new_width / 2.0, new_height / 2.0)
    dx, dy = (new_center[0] - center[0], new_center[1] - center[1])
    rot_mat[0, 2] += dx
    rot_mat[1, 2] += dy
    # img = cv2.warpAffine(img, rot_mat,(int(new_width), int(new_height)))
    return rot_mat, (int(new_width), int(new_height))


def show_sample(sample):
    img = sample['image']
    lm = sample['landmarks']
    plt.imshow(img, cmap='gray')
    plt.scatter(lm[:, 0], lm[:, 1], s=10, c='r')
    plt.show()


def test_rotation(img, lm, angle_degree):
    rot_mat, new_size = get_rotation_mat(img, angle_degree)
    new_img = cv2.warpAffine(img, rot_mat, new_size)
    lm = np.hstack((lm, np.ones((lm.shape[0], 1))))
    new_lm = np.dot(lm, rot_mat.T)
    plt.imshow(new_img, cmap='gray')
    plt.scatter(new_lm[:, 0], new_lm[:, 1], s=10, c='r')
    plt.show()


class RandomRotation(object):
    def __init__(self, angle_degree=10):
        self.angle = angle_degree

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        angle = np.random.uniform(-self.angle, self.angle, 1)
        rot_mat, new_size = get_rotation_mat(image, angle)

        image = cv2.warpAffine(image, rot_mat, new_size)
        landmarks = np.hstack((landmarks, np.ones((landmarks.shape[0], 1))))
        landmarks = np.dot(landmarks, rot_mat.T)

        return {'image': image, 'landmarks': landmarks}


class Normalize(object):
    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        image = (image / 255.0 - 0.5) / 0.5
        image = image[np.newaxis, ...]
        landmarks = (landmarks / 96.0 - 0.5) / 0.5
        return {'image': image, 'landmarks': landmarks}


class ToTensor(object):
    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        return {'image': torch.Tensor(image), 'landmarks': torch.Tensor(landmarks)}


class FaceDataset(Dataset):
    def __init__(self, pth_img, transform=None):
        self.images = np.load(pth_img)
        self.transform = transform
        self.num_imgs = len(self.images)

    def __len__(self):
        return self.num_imgs

    def __getitem__(self, idx):
        img = self.images[idx]
        lms = np.random.rand(30)
        sample = {'image': img, 'landmarks': lms.reshape(-1, 2)}
        if self.transform:
            sample = self.transform(sample)
        return sample['image']


class FaceLandmarkDataset(Dataset):
    def __init__(self, pth_img, pth_ldmk, transform=None):
        self.images = np.load(pth_img)
        self.transform = transform
        self.landmarks = np.load(pth_ldmk)
        self.num_imgs = len(self.images)

    def __len__(self):
        return self.num_imgs

    def __getitem__(self, idx):
        img = self.images[idx]
        lm = self.landmarks[idx]
        sample = {'image': img, 'landmarks': lm.reshape(-1, 2)}
        if self.transform:
            sample = self.transform(sample)
        return sample


def get_train_loader():
    scale = Rescale(110)
    crop = RandomCrop(110)
    flip = RandomFlip(0.5)
    rot = RandomRotation(15)
    norm = Normalize()
    totensor = ToTensor()
    composed = transforms.Compose([flip, scale, rot, crop, norm, totensor])

    train_data = FaceLandmarkDataset('train_data_total.npy', 'train_ldmk_total.npy', transform=composed)
    train_loader = Data.DataLoader(train_data, batch_size=32, shuffle=True, num_workers=4)
    return train_loader


def get_val_loader():
    scale = Rescale(110)
    norm = Normalize()
    totensor = ToTensor()
    composed = transforms.Compose([scale, norm, totensor])
    val_data = FaceLandmarkDataset('val_data.npy', 'val_ldmk.npy', transform=composed)
    val_loader = Data.DataLoader(val_data, batch_size=32, shuffle=True, num_workers=4)
    return val_loader


def get_test_loader():
    scale = Rescale(110)
    norm = Normalize()
    totensor = ToTensor()
    composed = transforms.Compose([scale, norm, totensor])
    test_data = FaceDataset('test_data.npy', transform=composed)
    test_loader = Data.DataLoader(test_data, batch_size=32, shuffle=False, num_workers=4)
    return test_loader


if __name__ == '__main__':
    train_data = FaceLandmarkDataset('train_data.npy', 'train_ldmk.npy')
    sample = train_data[0]
    img = sample['image']
    ldmk = sample['landmarks']

    scale = Rescale(110)
    crop = RandomCrop(110)
    flip = RandomFlip(0.5)
    rot = RandomRotation(10)
    composed = transforms.Compose([flip, scale, rot, crop])

    fig = plt.figure()
    for i, tsfrm in enumerate([scale, crop, flip, rot, composed]):
        transformed_sample = tsfrm(sample)
        img = transformed_sample['image']
        print(img.shape)
        lm = transformed_sample['landmarks']
        ax = plt.subplot(1, 5, i + 1)
        plt.tight_layout()
        ax.set_title(type(tsfrm).__name__)
        plt.imshow(img, cmap='gray')
        plt.scatter(lm[:, 0], lm[:, 1], s=10, c='r')

    plt.show()

    print('END')
