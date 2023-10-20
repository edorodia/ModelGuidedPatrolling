from PIL import Image
import numpy as np
from scipy import ndimage

# Load image from file
image = Image.open("Environment/Maps/mapa_tenerife_ndvi.png")
# Transform to grayscale
image = image.convert("L")
aspect_ratio = image.size[0] / image.size[1]
image = image.resize((int(aspect_ratio * 250), 250))
# Transform image to numpy array
image = np.array(image)
# Transform image to binary
#image[image < 128] = 0
#image[image >= 128] = 1
mask = np.genfromtxt("Environment/Maps/mapa_tenerife_mask.txt")
mask = ndimage.binary_erosion(mask, iterations=2)
image = (image/255.0) * mask
image[mask == 1] = 1 - image[mask == 1]
# Save image to file as integers
np.savetxt("Environment/Maps/mapa_tenerife_mask.txt", mask, fmt="%d")
np.savetxt("Environment/Maps/mapa_tenerife_ndvi.txt", image, fmt="%0.3f")

# Show image
import matplotlib.pyplot as plt
plt.imshow(image, cmap="gray")
plt.show()

