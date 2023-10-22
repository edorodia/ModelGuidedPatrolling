import numpy as np
import matplotlib.pyplot as plt
import torch as th
from torch.nn import AdaptiveAvgPool2d, AdaptiveMaxPool2d, MaxPool2d, AvgPool2d
from torch.nn.functional import max_pool2d, avg_pool2d
from torch.nn import Upsample, UpsamplingNearest2d, UpsamplingBilinear2d
from torchvision.transforms import GaussianBlur

# Load the dataset

trajectories = np.load(r'ModelTrain\Data\trajectories_algae_bloom_test.npy')
gts = np.load(r'ModelTrain/Data/gts_algae_bloom_test.npy')

plt.ion()

fig, ax = plt.subplots(1, 3, figsize = (10, 10))

d0 = ax[0].imshow(trajectories[0,0,0,:,:], vmin=0, vmax=1)

d1 = ax[1].imshow(trajectories[0,0,0,:,:], vmin=0, vmax=1)
d2 = ax[2].imshow(gts[0], vmin=0, vmax=1)

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
