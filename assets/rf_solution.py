from scipy.stats import pearsonr, zscore
import numpy as np
import matplotlib.pyplot as plt

# Load the data
stim = np.load(.....)
resp = np.load(.....)

num_samples = stim.shape[0]
height = stim.shape[1]
width = stim.shape[2]

# Method 1. Laborious method - loop over pixels, correlate timecourse, find max correlation
# Preallocate the correlation map as a 2D array of zeros
correlation_map = np.zeros((height, width))

# Loop over all pixels
for i in range(height):
    for j in range(width):

        # Grab one pixel's time series
        single_pixel_time_series = stim[:, i, j]
        
        # Compute the correlation between the pixel's time series and the response
        r, p = pearsonr(single_pixel_time_series, resp)
        
        # Store the correlation in the correlation map
        correlation_map[i,j] = r

# Find the index of the pixel with the highest correlation
max_correlation = np.max(correlation_map)
neuron_loc = np.where(correlation_map == max_correlation)
print('Method 1 (Pixelwise temporal correlations): The neuron with the highest correlation is at pixel', neuron_loc)

# Plot the correlation map
plt.imshow(correlation_map, cmap='gray', clim=[-0.5, 0.5])
plt.plot(neuron_loc[1], neuron_loc[0], 'ro', alpha=0.5, ms=20)
plt.title('Correlation map with neuron location at y={} and x={}'.format(neuron_loc[0], neuron_loc[1]))
plt.colorbar()
plt.show()

# Method 2. Nerdy way to do it that even the most seasoned Pythonistas might not know
# einsum is super useful and if I see someone using it, they instantly have my respect
# it's also a good introduction / application of vectorization (which you will want to start using
# instead of for loops). einsum isn't exactly vectorization, but a fancy way of 
# vectorizing that einstein invented. Basically, you're saying multiply each frame of the stimulus by each timepoint of the response then sum it all up. Because they are correlated, you should find one location that stands out. 
# Mathematically, this actually is almost identical to the first method.

receptive_field = np.einsum('nhw,n->hw', zscore(stim), zscore(resp))
print('Method 2 (Einstein Summation, essentially the Hardamard Product): the neuron with the highest correlation is at pixel', np.where(receptive_field==np.max(receptive_field)))

