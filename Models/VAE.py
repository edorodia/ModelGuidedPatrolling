import torch as th
import torch.nn as nn
from torchvision.models import vgg16_bn
from torchvision.transforms.functional import crop
# from torch import tensor as Tensor
from torch import nn


class VAE(nn.Module):

    """ Create a Variational Autoencoder with a fully connected architecture. """

    def __init__(self, input_size, latent_size, output_channels=1, loss_weights = None):


        """ Initialize the model. """

        super(VAE, self).__init__()

        self.input_size = input_size
        self.latent_size = latent_size
        
        if loss_weights is None:
            self.loss_weights = {'recon': 1, 'features': 1, 'kl': 1}
        else:
            self.loss_weights = loss_weights


        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(self.input_size[0], 16, 3, stride=1),
            nn.LeakyReLU(),
            nn.Dropout2d(0.5),
            nn.Conv2d(16, 32, 3, stride=1),
            nn.LeakyReLU(),
            nn.Dropout2d(0.5),
            nn.Conv2d(32, 64, 3, stride=1),
            nn.LeakyReLU(),
            #nn.Dropout2d(0.2),
            #nn.Conv2d(64, 128, 3, stride=1),
            #nn.LeakyReLU(),
        )

        # Get the number of features from the encoder
        with th.no_grad():
            self.encoder_size = th.tensor(self.encoder(th.zeros(1, *self.input_size)).shape[1:])
            # Get the number of features from the encoder
            self.encoder_size_out = th.prod(self.encoder_size)

        # Fully connected layers for the latent space        
        self.fc_logvar = nn.Linear(self.encoder_size_out, self.latent_size)
        self.fc_mu = nn.Linear(self.encoder_size_out, self.latent_size)
        self.fc_final = nn.Linear(self.latent_size, self.encoder_size_out)

        # Decoder - Exact Mirror of the Encoder
        self.decoder = nn.Sequential(
            #nn.ConvTranspose2d(128, 64, 3, stride=1, output_padding=1),
            #nn.LeakyReLU(),
            #nn.Dropout2d(0.2),
            nn.ConvTranspose2d(64, 32, 3, stride=1),
            nn.LeakyReLU(),
            nn.Dropout2d(0.5),
            nn.ConvTranspose2d(32, 16, 3, stride=1),
            nn.LeakyReLU(),
            nn.Dropout2d(0.5),
            nn.ConvTranspose2d(16, 8, 3, stride=1),
            nn.LeakyReLU(),
            # Final convolutional layers. Output shape should be the same as the input

            nn.Conv2d(8, 8, 2, padding=2, stride=1),
            nn.LeakyReLU(),
            nn.Dropout2d(0.5),
            nn.Conv2d(8, output_channels, 2),
            nn.Sigmoid(),
        )

        # Feature extractor
        self.feature_network = vgg16_bn(weights='IMAGENET1K_V1')
        self.feature_network.eval()
        # Freeze the feature network
        for param in self.feature_network.parameters():
            param.requires_grad = False


    def encode(self, x):

        """ Encode the input into a latent space. """

        x = self.encoder(x)
        x = x.view(-1, self.encoder_size_out)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def decode(self, z):
            
            """ Decode the latent space into the original input. """
            z = self.fc_final(z)
            z = z.view(-1, *self.encoder_size)
            return self.decoder(z)

    def reparameterize(self, mu, logvar):

        """ Reparameterize the latent space. """

        std = th.exp(0.5 * logvar)
        eps = th.randn_like(std)
        # eps = th.zeros_like(std)
        return mu + eps * std

    def forward(self, x):

        """ Encode and decode the input. """

        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return crop(self.decode(z), top=0, left=0, height=x.shape[2], width=x.shape[3]), mu, logvar

    def loss(self, x, x_hat, mu, logvar, mask=None):
        """ Compute the loss of the model. """

        # 1) Reconstruction loss - Only compute the loss on the masked pixels

        if mask is not None:
            mask = th.Tensor(mask).float().to(x.device)
            recon_loss = th.nn.functional.mse_loss(x_hat * mask, x * mask, reduction='sum') / mask.sum()
        else:
            recon_loss = th.nn.functional.mse_loss(x_hat, x, reduction='mean')

        # 2) KL divergence 
        kl_div = -0.5 * th.sum(1 + logvar - mu.pow(2) - logvar.exp())

        # 3) Feature loss between the output and the GT

        recons_features = self.extract_features(x)
        gt_features = self.extract_features(x_hat)

        feature_loss = 0.0
        for (r, i) in zip(recons_features, gt_features):
            feature_loss += th.nn.functional.mse_loss(r, i)

        # Compose the loss
        loss = self.loss_weights['recon'] * recon_loss + \
                self.loss_weights['kl'] * kl_div + \
                self.loss_weights['features'] * feature_loss

        return loss, self.loss_weights['recon'] * recon_loss, self.loss_weights['kl'] * kl_div, self.loss_weights['features'] * feature_loss

    def sample(self, n_samples):

        """ Sample from the latent space. """

        z = th.randn(n_samples, self.latent_size)

        return self.decode(z)

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
            feature_layers = ['14', '24', '34', '43']

        features = []
        # Triplicate the input if it is grayscale
        result = th.cat((input, input, input), dim=1)
        for (key, module) in self.feature_network.features._modules.items():
            result = module(result)
            if(key in feature_layers):
                features.append(result)

        return features


class EncoderDecoder(nn.Module):

    """ Create a Variational Autoencoder with a fully connected architecture. """

    def __init__(self, input_size, latent_size, output_channels=1, loss_weights = None):


        """ Initialize the model. """

        super(EncoderDecoder, self).__init__()

        self.input_size = input_size
        self.latent_size = latent_size
        
        if loss_weights is None:
            self.loss_weights = {'recon': 1, 'features': 1, 'kl': 1}
        else:
            self.loss_weights = loss_weights


        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(self.input_size[0], 16, 3, stride=1),
            nn.LeakyReLU(),
            nn.Dropout2d(0.5),
            nn.Conv2d(16, 32, 3, stride=1),
            nn.LeakyReLU(),
            nn.Dropout2d(0.5),
            nn.Conv2d(32, 64, 3, stride=1),
            nn.LeakyReLU(),
            #nn.Dropout2d(0.2),
            #nn.Conv2d(64, 128, 3, stride=1),
            #nn.LeakyReLU(),
        )

        # Get the number of features from the encoder
        with th.no_grad():
            self.encoder_size = th.tensor(self.encoder(th.zeros(1, *self.input_size)).shape[1:])
            # Get the number of features from the encoder
            self.encoder_size_out = th.prod(self.encoder_size)

        # Fully connected layers for the latent space        
        self.fc_mu = nn.Linear(self.encoder_size_out, self.latent_size)
        self.fc_final = nn.Linear(self.latent_size, self.encoder_size_out)

        # Decoder - Exact Mirror of the Encoder
        self.decoder = nn.Sequential(
            #nn.ConvTranspose2d(128, 64, 3, stride=1, output_padding=1),
            #nn.LeakyReLU(),
            #nn.Dropout2d(0.2),
            nn.ConvTranspose2d(64, 32, 3, stride=1),
            nn.LeakyReLU(),
            nn.Dropout2d(0.5),
            nn.ConvTranspose2d(32, 16, 3, stride=1),
            nn.LeakyReLU(),
            nn.Dropout2d(0.5),
            nn.ConvTranspose2d(16, 8, 3, stride=1),
            nn.LeakyReLU(),
            # Final convolutional layers. Output shape should be the same as the input

            nn.Conv2d(8, 8, 2, padding=2, stride=1),
            nn.LeakyReLU(),
            nn.Dropout2d(0.5),
            nn.Conv2d(8, output_channels, 2),
            nn.Sigmoid(),
        )

        # Feature extractor
        self.feature_network = vgg16_bn(weights='IMAGENET1K_V1')
        self.feature_network.eval()
        # Freeze the feature network
        for param in self.feature_network.parameters():
            param.requires_grad = False


    def encode(self, x):

        """ Encode the input into a latent space. """

        x = self.encoder(x)
        x = x.view(-1, self.encoder_size_out)
        mu = self.fc_mu(x)
        return mu

    def decode(self, z):
            
            """ Decode the latent space into the original input. """
            z = self.fc_final(z)
            z = z.view(-1, *self.encoder_size)
            return self.decoder(z)

    def reparameterize(self, mu, logvar):

        """ Reparameterize the latent space. """

        std = th.exp(0.5 * logvar)
        eps = th.randn_like(std)
        # eps = th.zeros_like(std)
        return mu + eps * std

    def forward(self, x):

        """ Encode and decode the input. """

        mu = self.encode(x)
        return crop(self.decode(mu), top=0, left=0, height=x.shape[2], width=x.shape[3])

    def loss(self, x, x_hat, mask=None):
        """ Compute the loss of the model. """

        # 1) Reconstruction loss - Only compute the loss on the masked pixels

        if mask is not None:
            mask = th.Tensor(mask).float().to(x.device)
            recon_loss = th.nn.functional.mse_loss(x_hat * mask, x * mask, reduction='sum') / mask.sum()
        else:
            recon_loss = th.nn.functional.mse_loss(x_hat, x, reduction='mean')


        # 3) Feature loss between the output and the GT

        #recons_features = self.extract_features(x)
        #gt_features = self.extract_features(x_hat)

        feature_loss = 0.0
        #for (r, i) in zip(recons_features, gt_features):
        #    feature_loss += th.nn.functional.mse_loss(r, i)

        # Compose the loss
        loss = self.loss_weights['recon'] * recon_loss # + self.loss_weights['features'] * feature_loss

        return loss, self.loss_weights['recon'] * recon_loss, self.loss_weights['features'] * recon_loss


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
            feature_layers = ['14', '24', '34', '43']

        features = []
        # Triplicate the input if it is grayscale
        result = th.cat((input, input, input), dim=1)
        for (key, module) in self.feature_network.features._modules.items():
            result = module(result)
            if(key in feature_layers):
                features.append(result)

        return features





if __name__ == '__main__':

    import matplotlib.pyplot as plt
    import numpy as np
    
    nav_map = np.genfromtxt('Environment/Maps/map.txt', delimiter=' ')
    nav_map = np.tile(nav_map, (2, 1, 1))

    x = th.tensor(nav_map).unsqueeze(0).float()

    vae = EncoderDecoder(nav_map.shape, 256, output_channels=1, loss_weights = None)

    vae.eval()

    x_hat = vae(x)

    print(x_hat.shape)
    print(x.shape)

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(x[0, 0].detach().numpy())
    ax[1].imshow(x_hat[0, 0].detach().numpy())
    plt.show()









