import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import cv2
import torch
import math


def get_grid(batch_img, batch_lm):
    processed = []
    for idx in range(batch_img.shape[0]):
        img = (batch_img[idx, 0].numpy() * 0.5 + 0.5) * 255.0
        img = cv2.cvtColor(img.astype('uint8'), cv2.COLOR_GRAY2BGR)
        lm = (batch_lm[idx].numpy().reshape((-1, 2)) * 0.5 + 0.5) * 96
        for lm_idx in range(lm.shape[0]):
            if not math.isnan(lm[lm_idx][0]):
                cv2.circle(img, (int(lm[lm_idx][0]), int(lm[lm_idx][1])), 3, (0, 255, 0), -1)
        processed.append(torch.from_numpy(img).permute(2, 0, 1))
    processed = torch.stack(processed, dim=0)
    I = make_grid(processed).permute(1, 2, 0)

    return I


def plot_grid(batch_img, batch_lm):
    I = get_grid(batch_img, batch_lm)
    plt.imshow(I.numpy().astype('uint8'))
    plt.show()
