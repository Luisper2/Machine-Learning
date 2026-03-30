import os
import numpy as np
from tqdm import tqdm
from skimage import io, color
import matplotlib.pylab as plt

os.system('cls')

# Problem 1
# Define the variables
trials = 100 # n Trials
n = 100      # Matrix size n x n
cache = []   # Save the Singular values for each iteration

# Iterate the process to achive the number of trials
for i in tqdm(range(trials)):
    matrix = np.random.randn(n, n)    # Generation of a random matrix using by default a normal distribution
    U, S, V = np.linalg.svd(matrix)  # Compute the svd of the matrix
    cache.append(S)                   # Saving of the Singular values

# Convertion cache to numpy
cache = np.array(cache)

# Plot of single values for all the trials
plt.figure(figsize=(12, 6))
plt.boxplot(cache.T, showfliers=False)
plt.title('Distribution of singular values')
plt.xlabel('Index')
plt.ylabel('Singular value')
ticks = [1] + list(range(10, 101, 10))
plt.xticks(ticks, [str(t) for t in ticks])
plt.savefig('Problem1.jpg')

# Problem 2
img = io.imread('image.jpg') # Load image
gray = color.rgb2gray(img)   # Convert it to gray scale

F = np.fft.fft2(gray)        # Compute FFT
F_shift = np.fft.fftshift(F) # Re-order of FFT

magnitude = np.abs(F_shift).ravel()   # Coumpute the magnitude
indices = np.argsort(magnitude)[::-1] # Sort index's

ratios = np.linspace(0.01, 1.0, 20) # Ratios sample
errors = []

norm_original = np.linalg.norm(gray) # Normalize gray image

for r in ratios:
    # Filter frequencies to keep
    k = int(r * len(indices))                 # n of coefficients to keep
    mask = np.zeros_like(F_shift, dtype=bool) # Create a mask
    flat_mask = mask.ravel()                  # Set all components as False
    flat_mask[indices[:k]] = True             # Mark as True the biggest coefficients
    mask = flat_mask.reshape(F_shift.shape)   # Re-shape

    F_compressed = np.where(mask, F_shift, 0) # Compressed version of FFT

    img_compressed = np.fft.ifft2(np.fft.ifftshift(F_compressed)).real # Reconstruct the image

    # Compute the error
    diff = gray - img_compressed
    error = (np.linalg.norm(diff) / norm_original) * 100
    errors.append(error)

# Plot of errors vs. ratios
plt.figure(figsize=(6,4))
plt.plot(ratios * 100, errors, marker='o')
plt.xlabel('Compression ratio (%)')
plt.ylabel('Error (%)')
plt.title('FFT Compression Error vs Ratio')
plt.grid(True)
plt.savefig('Problem2.jpg')