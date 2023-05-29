from PIL import Image
import numpy as np

# Load image from file
image = Image.open("Environment/Maps/mapa_ypacarai_detalle.png")
# Transform to grayscale
image = image.convert("L")
# Transform image to numpy array
image = np.array(image)
# Transform image to binary
image[image < 128] = 0
image[image >= 128] = 1
# Save image to file as integers
np.savetxt("Environment/Maps/map.txt", 1-image, fmt="%d")

# Show image
import matplotlib.pyplot as plt
plt.imshow(1-image, cmap="gray")
plt.show()

