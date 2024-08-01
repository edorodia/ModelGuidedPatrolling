import numpy as np
import matplotlib.pyplot as plt
import torch as th
from torch.nn import AdaptiveAvgPool2d, AdaptiveMaxPool2d, MaxPool2d, AvgPool2d
from torch.nn.functional import max_pool2d, avg_pool2d
from torch.nn import Upsample, UpsamplingNearest2d, UpsamplingBilinear2d
from torchvision.transforms import GaussianBlur

from mpl_toolkits.axes_grid1 import make_axes_locatable
import cmasher as cmr

# Load the dataset
#foldername = "ModelTrain/NoNoise_Shekel_Datasets/NoNoise_-_False_True_none"
foldername = "ModelTrain/Data"
trajectories = np.load(foldername + '/trajectories_shekel_train.npy', mmap_mode="r")
gts = np.load(foldername + '/gts_shekel_train.npy', mmap_mode="r")
background = np.genfromtxt('Environment/Maps/map.txt')

plt.ion()

cmap = cmr.get_sub_cmap('cmr.toxic', 0.30, 0.99)

fig, ax = plt.subplots(1, 3, figsize = (10, 10))

ax[0].imshow(background, vmin=0, vmax=1, cmap = 'copper_r', alpha = 1 - background, zorder=10)
d0 = ax[0].imshow(trajectories[0,0,0,:,:], vmin=0, vmax=1, cmap = 'gray')
ax[1].imshow(background, vmin=0, vmax=1, cmap = 'copper_r', alpha = 1 - background, zorder=10)
d1 = ax[1].imshow(trajectories[0,0,0,:,:], vmin=0, vmax=1, cmap = cmap)
ax[2].imshow(background, vmin=0, vmax=1, cmap = 'copper_r', alpha = 1 - background, zorder=10)

d2 = ax[2].imshow(gts[0], vmin=0, vmax=1,  cmap = cmap)

# Colorbar
divider = make_axes_locatable(ax[2])
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(d2, cax=cax)

plt.show()

def avg_pool(x):

    in_size = (x.shape[-2], x.shape[-1])
    # Check if tensor 
    if not isinstance(x, th.Tensor):
        x = th.Tensor(x).unsqueeze(0)
    
    return GaussianBlur((3,3))(x).squeeze().cpu().numpy()

for t in range(len(trajectories)):

    for i in range(trajectories.shape[1]):

        d0.set_data(trajectories[t,i,0,:,:]/255)
        d1.set_data(trajectories[t,i,1,:,:]/255)
        d2.set_data(gts[t]/255)
        fig.canvas.draw()
        plt.pause(0.1)

    input(  "Press Enter to continue..."    )
