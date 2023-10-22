import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.models import vgg16_bn

class DoubleConv(nn.Module):
    """(convolution => LeakyReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):

        return self.double_conv(x)
    
class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3] 

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return torch.sigmoid(self.conv(x))
    
class UNet(nn.Module):
    def __init__(self, n_channels_in, n_channels_out, bilinear=False, scale = 1):

        super(UNet, self).__init__()
        self.n_channels_in = n_channels_in
        self.n_channels_out = n_channels_out
        self.bilinear = bilinear
        self.scale = scale

        self.inc = (DoubleConv(n_channels_in, 64 // self.scale))
        self.down1 = (Down(64 // self.scale, 128 // self.scale))
        self.down2 = (Down(128 // self.scale, 256 // self.scale))
        self.down3 = (Down(256 // self.scale, 512 // self.scale))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512 // self.scale, 1024 // factor // self.scale))
        self.up1 = (Up(1024// self.scale, 512 // factor // self.scale, bilinear))
        self.up2 = (Up(512// self.scale, 256 // factor // self.scale, bilinear))
        self.up3 = (Up(256// self.scale, 128 // factor// self.scale, bilinear))
        self.up4 = (Up(128// self.scale, 64 // self.scale, bilinear))
        self.outc = (OutConv(64 // self.scale, n_channels_out))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
    
    def compute_loss(self, x_gt, x_predicted, mask = None):

        # Compute the loss

        if mask is not None:
            # Compute the loss only on the masked area
            loss = F.mse_loss(x_predicted * mask, x_gt * mask, reduction='sum') / mask.sum()
        else:
            loss = F.mse_loss(x_predicted, x_gt)

        return loss

class VAEUnet(nn.Module):

    def __init__(self, input_shape, n_channels_in, n_channels_out, bilinear=False, scale = 1):

        super(VAEUnet, self).__init__()

        
        # Create the encoder that will encode the latent space
        self.prior_encoder = nn.Sequential(
            nn.Conv2d(n_channels_in, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Flatten(),
        )

        prior_encoder_output_size = self.prior_encoder(torch.randn(1, n_channels_in, *input_shape)).shape[1]
        self.prior_encoder_mu = nn.Linear(prior_encoder_output_size, 64 // scale)
        self.prior_encoder_logvar = nn.Linear(prior_encoder_output_size, 64 // scale)

        # Create the decoder that will decode the latent space

        self.posterior_encoder =nn.Sequential(
            nn.Conv2d(n_channels_in + n_channels_out, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Flatten(),
        )

        self.input_shape = input_shape

        posterior_encoder_output_size = self.posterior_encoder(torch.randn(1, n_channels_in + n_channels_out, *input_shape)).shape[1]
        self.posterior_encoder_mu = nn.Linear(posterior_encoder_output_size, 64 // scale)
        self.posterior_encoder_logvar = nn.Linear(posterior_encoder_output_size, 64 // scale)


        self.n_channels_in = n_channels_in
        self.n_channels_out = n_channels_out
        self.bilinear = bilinear
        self.scale = scale

        self.inc = (DoubleConv(n_channels_in, 64 // self.scale))
        self.down1 = (Down(64 // self.scale, 128 // self.scale))
        self.down2 = (Down(128 // self.scale, 256 // self.scale))
        self.down3 = (Down(256 // self.scale, 512 // self.scale))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512 // self.scale, 1024 // factor // self.scale))
        self.up1 = (Up(1024// self.scale, 512 // factor // self.scale, bilinear))
        self.up2 = (Up(512// self.scale, 256 // factor // self.scale, bilinear))
        self.up3 = (Up(256// self.scale, 128 // factor// self.scale, bilinear))
        self.up4 = (Up(128// self.scale, 64 // self.scale, bilinear))
        self.outc = (OutConv(64 // self.scale, n_channels_out))

        # Feature extractor
        self.feature_network = vgg16_bn(weights='IMAGENET1K_V1')
        self.feature_network.eval()
        # Freeze the feature network
        for param in self.feature_network.parameters():
            param.requires_grad = False


    def sample(self, mu, logvar):
        """ Sample from the latent space using the channel wise reparametrization """

        # Obtain the standard deviation
        std = torch.exp(0.5*logvar)

        # Obtain the random noise
        eps = torch.randn(mu.shape[0], 64 // self.scale, *self.input_shape).to(mu.device)
        noise = torch.einsum('bchw, bc->bchw', eps, std)
        # Add the noise to the mean
        means = torch.einsum('bchw, bc->bchw', torch.ones_like(noise), mu)
        
        return means + noise

    def forward(self, x_in, x_gt, N = 1):
        """ Forward the model using the posterior network"""

        # Downsampling
        x1 = self.inc(x_in)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Encode using the posterior network

        # Compute the latent space of the posterior for training
        posterior = self.encode_posterior(torch.cat([x_in, x_gt], dim=1))
        # Obtain the mean and the log variance of the latent space
        

        # Upsampling
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        if N > 1:
            # Sample N times the latent space and create a tensor
            final_output = 0

            for i in range(N):
                # Sample the posterior
                z_posterior = self.sample(posterior[0], posterior[1])
                # Append the noise to the output and return the values of the last layer
                final_output = final_output + self.outc(x + z_posterior)

            # Compute the mean of the N samples
            output = final_output / N


        else:
            z_posterior = self.sample(posterior[0], posterior[1])
            output = self.outc(x + z_posterior)

        # Compute the latent space of the prior for training
        prior = self.encode_prior(x_in)

        return output, prior, posterior

    def forward_with_prior(self, x):
        """ Forward the model using the prior network """

        # Downsampling
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Compute the latent space
        prior = self.encode_prior(x)
        # Sample from the latent space
        z_prior = self.sample(prior[0], prior[1])

        # Upsampling
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        # Append the noise to the output and return the values of the last layer
        output = self.outc(x + z_prior)

        return output

    def encode_prior(self, x):

        # Compute the latent space
        x_hat = self.prior_encoder(x)
        mu_prior = self.prior_encoder_mu(x_hat)
        logvar_prior = self.prior_encoder_logvar(x_hat)

        return mu_prior, logvar_prior

    def encode_posterior(self, x):

        # Compute the latent space
        x_hat = self.posterior_encoder(x)
        mu_posterior = self.posterior_encoder_mu(x_hat)
        logvar_posterior = self.posterior_encoder_logvar(x_hat)

        return mu_posterior, logvar_posterior

    def compute_loss(self, x, x_true, x_out, prior, posterior, alpha = 1.0, beta = 0.01, gamma = 0.1, error_mask = None):

        # 1) Compute the reconstruction loss with the MSE
        if error_mask is not None:
            error_mask = torch.tile(error_mask, (len(x_out),1,1)).unsqueeze(1)
            reconstruction_loss = F.mse_loss(x_out[error_mask == 1], x_true[error_mask == 1]) 
        else:
            reconstruction_loss = F.mse_loss(x_out, x_true)

        # 2) Compute the KL divergence between the prior and the posterior
        
        q_posterior = torch.distributions.normal.Normal(posterior[0], torch.exp(posterior[1]))
        p_prior = torch.distributions.normal.Normal(prior[0], torch.exp(prior[1]))

        KL = torch.distributions.kl_divergence(q_posterior, p_prior).mean()

        # 3) Extract the features from the pretrained model
        features_true = self.extract_features(x_true)
        features_out = self.extract_features(x_out)
        # Compute the perceptual loss 
        feature_loss = 0.0
        for (r, i) in zip(features_out, features_true):
            feature_loss += torch.nn.functional.mse_loss(r, i)


        # 3) Compute the total loss

        loss = alpha * reconstruction_loss + beta * KL + gamma * feature_loss 

        return loss, reconstruction_loss, KL, feature_loss

    def extract_features(self,
                         input,
                         feature_layers = None):
        """
        Extracts the features from the pretrained model
        at the layers indicated by feature_layers.
        :param input: (Tensor) [B x C x H x W]
        :param feature_layers: List of string of IDs
        :return: List of the extracted features
        """
        if feature_layers is None:
            feature_layers = ['8', '12', '21', '28']

        features = []
        # Triplicate the input if it is grayscale
        result = torch.cat((input, input, input), dim=1)
        for (key, module) in self.feature_network.features._modules.items():
            result = module(result)
            if(key in feature_layers):
                features.append(result)

        return features

    def imagine(self, x, N):
        """ Use the prior to imagine N samples """

        # Compute the latent space
        # Downsampling
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Compute the latent space
        prior = self.encode_prior(x)

        # Upsampling
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        # Sample N times the latent space and create a tensor 
        output = torch.zeros((N, 1, *self.input_shape)).to(x.device)

        for i in range(N):
            # Sample the prior
            z_prior = self.sample(prior[0], prior[1])
            # Append the noise to the output and return the values of the last layer
            output[i,:,:] = self.outc(x + z_prior)

        return output.mean(dim=0), output.std(dim=0)




if __name__ == "__main__":

    import matplotlib.pyplot as plt

    model = VAEUnet((60, 59), 2, 1)

    input_tensor = torch.randn(1, 2, 60, 59)
    ground_truth_tensor = torch.randn(1, 1, 60, 59)

    output_tensor = model(input_tensor, ground_truth_tensor)

    print(output_tensor.shape)
    print(input_tensor.shape)

    fig, ax = plt.subplots(1, 2)

    ax[0].set_title("Input")
    ax[0].imshow(input_tensor[0, 0, :, :].detach().numpy())
    ax[1].set_title("Output")
    ax[1].imshow(output_tensor[0, 0, :, :].detach().numpy())
    plt.show()

    