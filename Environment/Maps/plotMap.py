import numpy as np
import matplotlib.pyplot as plt

# Create a 2D matrix (example: 10x10 matrix with random values)
matrix = np.genfromtxt('map.txt', delimiter=' ')

# Create a plot
plt.figure(figsize=(8, 6))

# Use imshow to display the matrix
plt.imshow(matrix, cmap='viridis', interpolation='nearest')

# Add a colorbar to show the scale
plt.colorbar(label='Intensity')

# Add title and labels
plt.title('2D Matrix Plot')
plt.xlabel('Column Index')
plt.ylabel('Row Index')

# Display the plot
plt.show()
