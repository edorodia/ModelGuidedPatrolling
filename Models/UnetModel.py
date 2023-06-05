import sys
sys.path.append('.')
import torch as th
import numpy as np
import matplotlib.pyplot as plt
from Models.unet import UNet
from Models.MiopicModel import MiopicModel


benchmark_2_path = {'algae_bloom': 'runs/TrainingUnet/Unet_algae_bloom_20230530-001951/Unet_algae_bloom_train.pth',
                    'shekel': 'runs/TrainingUnet/Unet_shekel_20230530-004507/Unet_shekel_train.pth',}

class UnetDeepModel:

    def __init__(self, navigation_map: np.ndarray, model_path: str, device: str = 'cuda:0', resolution = 1, influence_radius = 2, dt = 0.7):

        self.navigation_map = navigation_map
        self.device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')

        # Create the miopic predictor
        self.pre_model = MiopicModel(navigation_map, influence_radius, resolution, dt)
        # Create the model
        self.model = UNet(n_channels_in=2, n_channels_out=1, bilinear=False, scale=2).to(device)
        # Import the model
        self.model.load_state_dict(th.load(model_path))
        self.model.eval()

        self.model_map = np.zeros_like(self.navigation_map)

    def update(self, x: np.ndarray, y: np.ndarray, t: np.ndarray = None):

        # Update the miopic model
        self.pre_model.update(x, y)

        # Use the miopic model to predict the map
        pre_model_map = self.pre_model.predict()

        with th.no_grad():

            # Feed the model
            input_tensor_0 = th.from_numpy(t).unsqueeze(0).unsqueeze(0).to(self.device).float()
            input_tensor_1 = th.from_numpy(pre_model_map).unsqueeze(0).unsqueeze(0).to(self.device).float()
            # Stack using the dim 1
            input_tensor = th.cat((input_tensor_0, input_tensor_1), dim=1)
            # Predict the model #
            output_tensor = self.model(input_tensor)
            # Get the numpy array
            self.model_map = output_tensor.squeeze(0).squeeze(0).cpu().detach().numpy() * self.navigation_map

    def predict(self):

        return self.model_map

    def reset(self):

        self.model_map = np.zeros_like(self.navigation_map)
        self.pre_model.reset()
