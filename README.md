# P1 Coding Assessment

## Table of Contents
* [Problem Statement](#problem)
* [High-Level Solution Methodology](#methodology)
* [Reason for selecting KMeans Clustering](#reasons)
* [Data Exploration](#exploration)
* [Applying Fourier Transform to Obtain Frequency Spectrum of Each Signal](#fourier)
* [Apply KMeans Clustering to Spectra](#kmeans)
* [Color coding the spectra according to their cluster](#color)
* [Plotting clusters against the dominant frequency bands](#plot)

### Problem Statement <a name='problem'></a>

A zip file was provided with 40 files - all of them `.npy` object files (Numpy objects). The following instructions were provided:

Use a clustering method, to cluster the attached data into 3 groups.

__Dataset:__
* each data sample is a 1D signal containing 2400 points
 
__Expected Results:__
* What is the method used for the clustering? How is it decided which method to use?
* Visualization of the clusters with the labels of the data point
* The developed codes

__Hint:__ You may need to focus on a part of the signal for the best clustering

The file names were as follows: 

```
 '2018-10-15_11-31-41-PASSTHROUGH-12-59-31.npy',
 '2018-10-18_14-25-40-PASSTHROUGH-12-59-31.npy',
 '2018-10-15_11-30-03-PASSTHROUGH-12-59-31.npy',
 '2018-10-15_11-40-25-PASSTHROUGH-12-59-31.npy',
 '2018-10-18_14-20-19-PASSTHROUGH-12-59-31.npy',
 '2018-10-15_11-55-18-PASSTHROUGH-12-59-31.npy',
 '2018-10-18_14-10-19-PASSTHROUGH-12-59-31.npy',
 '2018-10-15_11-36-36-PASSTHROUGH-12-59-31.npy',
 '2018-10-15_11-53-51-PASSTHROUGH-12-59-31.npy',
 '2018-10-18_14-56-57-PASSTHROUGH-12-59-31.npy',
 '2018-10-18_14-06-45-PASSTHROUGH-12-59-31.npy',
 '2018-10-18_14-29-51-PASSTHROUGH-12-59-31.npy',
 '2018-10-15_12-02-56-PASSTHROUGH-12-59-31.npy',
 '2018-10-15_11-38-01-PASSTHROUGH-12-59-31.npy',
 '2018-10-18_14-07-48-PASSTHROUGH-12-59-31.npy',
 '2018-10-18_14-30-51-PASSTHROUGH-12-59-31.npy',
 '2018-10-18_14-00-47-PASSTHROUGH-12-59-31.npy',
 '2018-10-15_11-44-00-PASSTHROUGH-12-59-31.npy',
 '2018-10-18_13-58-38-PASSTHROUGH-12-59-31.npy',
 '2018-10-18_14-19-14-PASSTHROUGH-12-59-31.npy',
 '2018-10-15_11-48-43-PASSTHROUGH-12-59-31.npy',
 '2018-10-15_11-39-18-PASSTHROUGH-12-59-31.npy',
 '2018-10-15_11-45-03-PASSTHROUGH-12-59-31.npy',
 '2018-10-18_14-32-01-PASSTHROUGH-12-59-31.npy',
 '2018-10-18_14-26-39-PASSTHROUGH-12-59-31.npy',
 '2018-10-15_11-56-44-PASSTHROUGH-12-59-31.npy',
 '2018-10-18_14-54-45-PASSTHROUGH-12-59-31.npy',
 '2018-10-15_12-05-27-PASSTHROUGH-12-59-31.npy',
 '2018-10-18_14-35-02-PASSTHROUGH-12-59-31.npy',
 '2018-10-18_14-18-15-PASSTHROUGH-12-59-31.npy',
 '2018-10-18_14-55-53-PASSTHROUGH-12-59-31.npy',
 '2018-10-18_14-38-29-PASSTHROUGH-12-59-31.npy',
 '2018-10-18_14-13-11-PASSTHROUGH-12-59-31.npy',
 '2018-10-18_14-05-15-PASSTHROUGH-12-59-31.npy',
 '2018-10-18_14-24-33-PASSTHROUGH-12-59-31.npy',
 '2018-10-18_14-02-47-PASSTHROUGH-12-59-31.npy',
 '2018-10-18_14-11-51-PASSTHROUGH-12-59-31.npy',
 '2018-10-15_12-04-10-PASSTHROUGH-12-59-31.npy',
 '2018-10-15_11-33-03-PASSTHROUGH-12-59-31.npy',
 '2018-10-18_14-36-01-PASSTHROUGH-12-59-31.npy'
```

### High-Level Solution Methodology <a name='methodology'></a>

My approach to solving this problem was as follows: 

* Window the time series signal to the active components.
* Get frequency domain representation for each of the 40 windowed signals.
* Fit a KMeans clustering algorithm to the 40 frequency spectra using 3 clusters.  

### Reasoning for Selecting KMeans <a name='reasons'></a>

At a high-level, I looked at using three types of clustering algorithms:

* Hierarchical 
* Density-based 
* KMeans   

__Hierarchical clustering__ could be a decent alternative in this application, but is best suited to finding embedded structure within clusters. Hierarchical clustering is great when all you know is that your data can likely be subdivided into groups, but there is additional interest in the substructure of those groups.  Since this problem explicitly asks for three clusters, hierarchical clustering seemed like an inferior choice to an explicit means-based clustering method.

__Density-based clustering__ looks at tightly-packed groups of data and assumes everything else is noise.  Because I chose to cluster the frequency spectra data, this method isn't really appropriate because the density variation on a 1-D or 2-D scale is similar, if not the same.  The other pitfall here is that density based algorithms assume data far from clusters are just noise.  In this application we are given individual signals to clusters - each signal is presumably meaningful and represents something meaningful.  Because of this, treating outliers as noise doesn't seem appropriate.  

__Kmeans clustering__ was ultimately arrived at for the following reasons:
* Where hierarchical discovers embedded structure and density-based methods excel at finding unknown numbers of clusters with similar density, both of these methods do a poor job of reaching a cluster "consensus" in a full dataset.  KMeans will divide your data into N-groups even if your data is uniform (which is a benefit and a pitfall depending on the application).
* KMeans considers every point in the dataset as meaningful.  
* Our frequency data has a consistent amplitude profile for each signal, which reduces the chances of KMeans "trapping" itself in a local minima.  
* Our data is not sparse.  
* It is clear from the frequency data that the dominant distinction between the signals is the amplitudes around the lower band.  Kmeans should perform well in seperating these for us.  


### Data Exploration <a name='exploration'></a>

The first thing I decided to do was to visualize the data provided in each `.npy` file.  Visually, the signals appear to be roughly the same type - my suspicion is that they're intended to mimic some sort of radio or radar reflection signal.  To visualize the raw signals I did the following: 

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp
import scipy.fftpack
from os import listdir

#Set figure size for matplotlib
plt.rcParams['figure.figsize'] = [20, 12]
plt.rcParams['axes.titlesize'] = 24
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16

# Loop through, load, and plot .npy files
for file in files:
    x = np.load(file)
    plt.plot(x)
plt.title('Raw data plotted on same scale')
plt.show()
```

<br>

![raw_data](/images/raw_data.png)

<br>
<br>
Because the body of the signals begins and ends in roughly the same location, I decided to trim them from index 125 to index 800 to arrive at the following.

<br>
<br>

![windowed_raw_data](/images/windowed_raw.png)

<br>
<br>

### Applying Fourier Transform to Obtain Frequency Spectrum of Each Signal <a name='fourier'></a>

The next step in the process was to convert each signal to the frequency domain which can then be used in the KMeans clustering algorithm.  This was achieved as follows:

```python

freqs = sp.fftpack.fftfreq(675)
spectra = []
fig, ax = plt.subplots()

# Because frequency domain is symetrical, take only positive frequencies
i = freqs > 0
for signal in data:
    X = sp.fftpack.fft(signal)  
    ax.plot(freqs[i], np.abs(X)[i])
    spectra.append(np.abs(X)[i])

ax.set_xlabel('Frequency')
ax.set_ylabel('Frequency Domain (Spectrum) Magnitude')
ax.set_title('Frequency Domain Representation of Each Signal')
```

<br>
<br>

![frequency_domain](/images/frequency_domain.png)

<br>
<br>

### Apply KMeans Clustering Algorithm <a name='kmeans'></a>

The next thing I did was to apply the KMeans clustering algorithm to cluster the data into three groups.  I 

```python
from sklearn.cluster import KMeans
kmeans = KMeans(3, max_iter = 1000, n_init = 100)
kmeans.fit_transform(spectra)
predictions = kmeans.predict(spectra)
predictions
```

Output:
```
array([1, 1, 2, 1, 0, 1, 2, 1, 2, 0, 0, 2, 2, 1, 0, 1, 0, 2, 2, 0, 0, 2,
       1, 2, 0, 1, 2, 0, 2, 2, 0, 1, 0, 2, 2, 0, 0, 1, 2, 1], dtype=int32)
```

### Color coding the spectra according to their cluster <a name='color'></a>

By re-plotting each spectra with a color coding according to its cluster, we can see the differentiation start to appear when all of the spectra are plotted together.  

```python
fig2, ax2 = plt.subplots()
for spectra_id, color in enumerate(['red','blue','green']):
    mask = list(np.where(predictions==spectra_id)[0])
    print(mask)
    for elem in mask:
        ax2.plot(freqs[i], spectra[elem], color=color)

ax2.set_xlabel('Frequency')
ax2.set_ylabel('Frequency Domain (Spectrum) Magnitude')
```

<br>
<br>

![frequency_clustered](/images/frequency_clustered.png)

<br>
<br>

### Plotting clusters against the dominant frequency bands <a name='plot'></a>

The next thing I want to do is plot each of these spectra on a two dimensional plot where each dimension is one of the two dominant frequency bands - roughly __0.02 - 0.08__ (band1) and __0.14-0.18__ (band2).

```python
positive_frequencies = freqs[i]

# Create masks for the spectra that filter all but each band.
band1 = np.where(np.logical_and(positive_frequencies>=0.02, positive_frequencies<=0.08))
band2 = np.where(np.logical_and(positive_frequencies>=0.14, positive_frequencies<=0.18))

x = []
y = [] 

# For each spectrum, find the mean amplitude within each band.
for spectrum in spectra:
    x.append(np.mean(spectrum[band1]))
    y.append(np.mean(spectrum[band2])) 

fig3, ax3 = plt.subplots()
for spectra_id, color in enumerate(['red','blue','green']):
    mask = list(np.where(predictions==spectra_id)[0])
    for elem in mask:
        ax3.scatter(x[elem], y[elem], color=color)
        ax3.text(x[elem]+0.00003, y[elem], elem, fontsize=12)

ax3.set_ylim(0.000, 0.0025)
ax3.set_xlim(0.0005, 0.0045)
ax3.set_xlabel('Mean Frequency Amplitute in band 0.02-0.08')
ax3.set_ylabel('Mean Frequency Amplitute in band 0.14-0.18')
ax3.set_title('Mean Signal Amplitude by Band1 and Band2') 
```

<br>
<br>

![frequency_clustered](/images/scatter.png)

<br>
<br>
The indexes of these spectra correspond to the files names accordingly:

```
(0,  '2018-10-15_11-31-41-PASSTHROUGH-12-59-31.npy')
(1,  '2018-10-18_14-25-40-PASSTHROUGH-12-59-31.npy')
(2,  '2018-10-15_11-30-03-PASSTHROUGH-12-59-31.npy')
(3,  '2018-10-15_11-40-25-PASSTHROUGH-12-59-31.npy')
(4,  '2018-10-18_14-20-19-PASSTHROUGH-12-59-31.npy')
(5,  '2018-10-15_11-55-18-PASSTHROUGH-12-59-31.npy')
(6,  '2018-10-18_14-10-19-PASSTHROUGH-12-59-31.npy')
(7,  '2018-10-15_11-36-36-PASSTHROUGH-12-59-31.npy')
(8,  '2018-10-15_11-53-51-PASSTHROUGH-12-59-31.npy')
(9,  '2018-10-18_14-56-57-PASSTHROUGH-12-59-31.npy')
(10, '2018-10-18_14-06-45-PASSTHROUGH-12-59-31.npy')
(11, '2018-10-18_14-29-51-PASSTHROUGH-12-59-31.npy')
(12, '2018-10-15_12-02-56-PASSTHROUGH-12-59-31.npy')
(13, '2018-10-15_11-38-01-PASSTHROUGH-12-59-31.npy')
(14, '2018-10-18_14-07-48-PASSTHROUGH-12-59-31.npy')
(15, '2018-10-18_14-30-51-PASSTHROUGH-12-59-31.npy')
(16, '2018-10-18_14-00-47-PASSTHROUGH-12-59-31.npy')
(17, '2018-10-15_11-44-00-PASSTHROUGH-12-59-31.npy')
(18, '2018-10-18_13-58-38-PASSTHROUGH-12-59-31.npy')
(19, '2018-10-18_14-19-14-PASSTHROUGH-12-59-31.npy')
(20, '2018-10-15_11-48-43-PASSTHROUGH-12-59-31.npy')
(21, '2018-10-15_11-39-18-PASSTHROUGH-12-59-31.npy')
(22, '2018-10-15_11-45-03-PASSTHROUGH-12-59-31.npy')
(23, '2018-10-18_14-32-01-PASSTHROUGH-12-59-31.npy')
(24, '2018-10-18_14-26-39-PASSTHROUGH-12-59-31.npy')
(25, '2018-10-15_11-56-44-PASSTHROUGH-12-59-31.npy')
(26, '2018-10-18_14-54-45-PASSTHROUGH-12-59-31.npy')
(27, '2018-10-15_12-05-27-PASSTHROUGH-12-59-31.npy')
(28, '2018-10-18_14-35-02-PASSTHROUGH-12-59-31.npy')
(29, '2018-10-18_14-18-15-PASSTHROUGH-12-59-31.npy')
(30, '2018-10-18_14-55-53-PASSTHROUGH-12-59-31.npy')
(31, '2018-10-18_14-38-29-PASSTHROUGH-12-59-31.npy')
(32, '2018-10-18_14-13-11-PASSTHROUGH-12-59-31.npy')
(33, '2018-10-18_14-05-15-PASSTHROUGH-12-59-31.npy')
(34, '2018-10-18_14-24-33-PASSTHROUGH-12-59-31.npy')
(35, '2018-10-18_14-02-47-PASSTHROUGH-12-59-31.npy')
(36, '2018-10-18_14-11-51-PASSTHROUGH-12-59-31.npy')
(37, '2018-10-15_12-04-10-PASSTHROUGH-12-59-31.npy')
(38, '2018-10-15_11-33-03-PASSTHROUGH-12-59-31.npy')
(39, '2018-10-18_14-36-01-PASSTHROUGH-12-59-31.npy')
```