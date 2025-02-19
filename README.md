# GEOL0069_Week4 Assessment: Satellite Echo Classification

This repository aims to classify the echoes in leads and sea ice by machine learning, generate the average echo shape and standard deviation for both classes, and quantify echo classification according to the official ESA classification using the confusion matrix.


<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#Colocating Sentinel-3 OLCI/SRAL and Sentinal-2 Optical Data">Colocating Sentinel-3 OLCI/SRAL and Sentinal-2 Optical Data</a>
      <ul>
        <li><a href="#Step 0: Read in Functions Needed">Step 0: Read in Functions Needed</a></li>
        <li><a href="#Step 1: Get the Metadata for satellites (Sentinel-2 and Sentinel-3 OLCI in this case)">Step 1: Get the Metadata for satellites (Sentinel-2 and Sentinel-3 OLCI in this case)</a></li>
        <ul>
          <li><a href="#Co-locate the data">Co-locate the data</a></li>
        <li><a href="#Proceeding with Sentinel-3 OLCI Download">Proceeding with Sentinel-3 OLCI Download</a></li>
           <li><a href="#Sentinel-3 SRAL">Sentinel-3 SRAL</a></li>
      </ul>
        </ul>
    </li>
    <li>
     <a href="#Unsupervised Learning">Unsupervised Learning</a>
      <ul>
        <li><a href="#Introduction to Unsupervised Learning Methods [Bishop and Nasrabadi, 2006]">Introduction to Unsupervised Learning Methods [Bishop and Nasrabadi, 2006]</a></li>
        <ul>
          <li><a href="#Introduction to K-means Clustering">Introduction to K-means Clustering</a></li>
        <li><a href="#Why K-means for Clustering?">Why K-means for Clustering?</a></li>
        <li><a href="#Key Components of K-means">Key Components of K-means</a></li>
        <li><a href="#The Iterative Process of K-means">The Iterative Process of K-means</a></li>
        <li><a href="#Advantages of K-means">Advantages of K-means</a></li>
        <li><a href="#Basic Code Implementation">Basic Code Implementation</a></li>
          </ul>
        <li><a href="#Gaussian Mixture Models (GMM) [Bishop and Nasrabadi, 2006]">Gaussian Mixture Models (GMM) [Bishop and Nasrabadi, 2006]</a></li>
        <ul>
        <li><a href="#Introduction to Gaussian Mixture Models">Introduction to Gaussian Mixture Models</a></li>
        <li><a href="#Why Gaussian Mixture Models for Clustering?">Why Gaussian Mixture Models for Clustering?</a></li>
        <li><a href="#Key Components of GMM">Key Components of GMM</a></li>
        <li><a href="#The EM Algorithm in GMM">The EM Algorithm in GMM</a></li>
        <li><a href="#Advantages of GMM">Advantages of GMM</a></li>
        <li><a href="#Basic Code Implementation">Basic Code Implementation</a></li>
           </ul>
        <li><a href="#Image Classification">Image Classification</a></li>
        <ul>
        <li><a href="#K-Means Implementation">K-Means Implementation</a></li>
        <li><a href="#GMM Implementation">GMM Implementation</a></li>
           </ul>
        <li><a href="#Altimetry Classification">Altimetry Classification</a></li>
        <ul>
        <li><a href="#Read in Functions Needed">Read in Functions Needed</a></li>
          </ul>
        <li><a href="#Scatter Plots of Clustered Data">Scatter Plots of Clustered Data</a></li>
        <li><a href="#Waveform Alignment Using Cross-Correlation">Waveform Alignment Using Cross-Correlation</a></li>
        <li><a href="#Compare with ESA data">Compare with ESA data</a></li>   
       
  
  </ol>
</details>


<!-- ABOUT THE PROJECT -->
## About The Project

This project explores the collocation of Sentinel-3 data and Sentinel-2 optical data, while using unsupervised learning techniques to classify sea ice and leads. The goal is to classify environmental features by fusing different satellite datasets and applying machine learning models, providing a clear understanding and effective tools for using unsupervised learning methods in real world (EO) scenarios.

Here's why:

* This fusion of datasets provides a richer, more detailed perspective of Earth's surface.
* Using machine learning to identifying patterns and categorising data in EO data.


Of course, the application of AI is a growing field, and approaches to analysing satellite data through AI abound. This project is a practical guide combining sensing satellite data and AI machine learning. Your contributions and improvements are welcome!

<p align="right">(<a href="#readme-top">back to top</a>)</p>


### Built With

This project utilises key Python libraries and geospatial tools to process, analyse, and classify Earth Observation (EO) data. Below are the major dependencies used:

* NumPy
* Shapely
* Datatime
* Requests
* Pandas
* OS
* Json
* Folium
* Xml.etree
* IPython
* Rasterio
* Sklearn
* NetCDF4
* Glob
* Scipy
* Matplotlib

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- Colocating Sentinel-3 OLCI/SRAL and Sentinal-2 Optical Data -->
## Colocating Sentinel-3 OLCI/SRAL and Sentinal-2 Optical Data
This section explores in detail the collocation of Sentinel-3 data with Sentinel-2 optical data. Data synchronisation from the two satellite missions enables powerful synergies, leveraging Sentinel-2's high spatial resolution and Sentinel-3's comprehensive coverage and synchronisation of altimeter data. This fusion of data sets provides a richer and more detailed view of the Earth's surface. The following sections show the steps necessary to complete the identification and alignment of these data sets.

<!-- Step 0: Read in Functions Needed -->
### Step 0: Read in Functions Needed

To simplify data acquisition and processing, first load basic functions to help obtain metadata for both satellites. Start by making sure that Google Drive is mounted in Google Colab for access to files.
```python
from google.colab import drive
drive.mount('/content/drive')
```
Using NumPy, shapely, datatime, requests, pandas, xml.etree, os, json and folium, scripts fetch, process, and visualise data from the Copernicus Data Space Ecosystem. The specific codes are in the code folder -- Data Colocating.

```python
from datetime import datetime, timedelta
from shapely.geometry import Polygon, Point, shape
import numpy as np
import requests
import pandas as pd
from xml.etree import ElementTree as ET
import os
import json
import folium
```

<!-- Step 1: Get the Metadata for satellites (Sentinel-2 and Sentinel-3 OLCI in this case) -->
### Step 1: Get the Metadata for satellites (Sentinel-2 and Sentinel-3 OLCI in this case)
We illustrate how we co-locate Sentinel-2 and Sentinel-3 OLCI by first obtaining metadata. Since we were trying to find colocation between 2 satellites, we obtained 2 metadata tables representing 2 satellites of concern to us. In this case, they are named objects' s3_olci_metadata 'and' s2_metadata '. And using IPython to display those data.

```python
from IPython.display import display

display(s3_olci_metadata)
```
![image](https://github.com/Guoxuan-Li/GEOL0069_Week4/blob/main/Images/s3_olci_metadata.png?raw=true)

```python
from IPython.display import display

display(s2_metadata)
```
![image](https://github.com/Guoxuan-Li/GEOL0069_Week4/blob/main/Images/s2_metadata.png?raw=true)

These tables display metadata for Sentinel-3 OLCI images and Sentinel-2 images by using the Copernicus Data Space API.

<!-- Co-locate the data -->
#### Co-locate the data
Using these metadata to co-locate them and produce another table shows the details of the colocation pairs, and match rows from S2 and S3 OLCI by their geo_footprint and save the results as 'results'. The top 5 results should like the table below:

![image](https://github.com/Guoxuan-Li/GEOL0069_Week4/blob/main/Images/results.head(5).png?raw=true)

In addition, we can use code to visualise the co-located footprint from Sentinel-2 and Sentinel-3 OLCI.
```python
from IPython.display import display

map_result = plot_results(results.head(5))
display(map_result)
```
![image](https://github.com/Guoxuan-Li/GEOL0069_Week4/blob/main/Images/map_result1.jpg?raw=true)


<!-- Proceeding with Sentinel-3 OLCI Download -->
#### Proceeding with Sentinel-3 OLCI Download
Download the Sentinel-3 OLCI data, and keep the method consistent with Sentinel-2. We will apply the same filename conversion logic and follow the structured steps to retrieve data from the Copernican data space.

<!-- Sentinel-3 SRAL -->
#### Sentinel-3 SRAL
It is also possible to co-locate S2/S3 OLCI with S3 SRAL (altimetry data). We just need to fetch the S3 SRAL metadata and do the co-location again for S3 SRAL with S2, and visualise the co-location results.

![image](https://github.com/Guoxuan-Li/GEOL0069_Week4/blob/main/Images/map_result2.jpg?raw=true)

The blue outline represents the geographic footprint of the two satellite datasets detected in the overlapping area of the Arctic region.


<!-- Unsupervised Learning -->
## Unsupervised Learning

This section will use unsupervised learning to perform a classification task for Sea ice and lead in the real-world (EO) scenarios. This is because unsupervised learning techniques are particularly good at identifying patterns and classifying data when categories are not explicitly labeled. By exploring these techniques, we will gain insight into how to identify structures and relationships in datasets.

<!-- Introduction to Unsupervised Learning Methods [Bishop and Nasrabadi, 2006] -->
### Introduction to Unsupervised Learning Methods [Bishop and Nasrabadi, 2006]

<!-- Introduction to K-means Clustering -->
#### Introduction to K-means Clustering

K-means clustering is an unsupervised learning algorithm used to divide a data set into a set of k groups (or clusters), where k represents the number of groups pre-specified by the analyst. It classifies data points based on the similarity of the data {cite}macqueen1967some features. The basic idea is to define k centers of mass, one for each cluster, and then assign each data point to the nearest center of mass while keeping the center of mass as small as possible.

<!-- Why K-means for Clustering? -->
#### Why K-means for Clustering?
K-means clustering is effective when the structure of the data is not known beforehand, and it is simplicity adn scalability. K-means doesn't require any prior knowledge of data distribution or structure, making it ideal for exploratory data analysis. And the algorithm is easy to implement and can be extended to large datasets relatively easily.

<!-- Key Components of K-means -->
#### Key Components of K-means
There are four key components of K-means: choosing K, centroids initialisation, assignment step and update step. Specify the number of clusters (K) before applying the algorithm, initialise the centroid position to avoid affecting the final result, assign each data point to the nearest centroid based on the square of the Euclidean distance and finally recalculate the centroid as the center of all data points assigned to the respective clusters.

<!-- The Iterative Process of K-means -->
#### The Iterative Process of K-means
The assignment and update steps are repeated iteratively until the center of mass no longer moves significantly, which means that intra-cluster changes are minimised. This iterative process ensures that the algorithm converges to a result that may be locally optimal.

<!-- Advantages of K-means -->
#### Advantages of K-means
K-means is computationally efficient and ease of interpretation.

<!-- Basic Code Implementation -->
#### Basic Code Implementation
This section provides a K-means clustering implementation. Connect to colab and Google drive as before, then pip the rasterio and netCDF4 to read and process the data.

```python
from google.colab import drive
drive.mount('/content/drive')
```
```python
pip install rasterio
```
```python
pip install netCDF4
```

Using sklearn, created 100 data points randomly, the K-means model is initialised, and assigned each data point. Visualise the data as a scatter plot, with color-coded points representing clusters, and the calculated centroid marked with a black dot. This illustrates how K-means groups data for pattern recognition and segmentation.

```python
# Python code for K-means clustering
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

# Sample data
X = np.random.rand(100, 2)

# K-means model
kmeans = KMeans(n_clusters=4)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

# Plotting
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
plt.show()
```
<img src="https://github.com/Guoxuan-Li/GEOL0069_Week4/blob/main/Images/K_means.jpg?raw=true" width="500" height="400" />


<!-- Gaussian Mixture Models (GMM) [Bishop and Nasrabadi, 2006] -->
### Gaussian Mixture Models (GMM) [Bishop and Nasrabadi, 2006]

<!-- Introduction to Gaussian Mixture Models -->
#### Introduction to Gaussian Mixture Models
Gaussian mixture model (GMM) is a probabilistic model used to represent normally distributed subpopulations in a population. The model assumes that the data is generated from a mixture of several Gaussian distributions, each with its own mean and variance[Reynolds et al., 2009]. GMM is widely used in clustering and density estimation because they provide a way to represent complex distributions by combinations of simple distributions.

<!-- Why Gaussian Mixture Models for Clustering? -->
#### Why Gaussian Mixture Models for Clustering?
Gaussian mixture model is useful where soft clusering is needed and flexibility in cluster covariance. GMM provides the probability that each data point belongs to each cluster, providing a soft classification and understanding of data uncertainty. And it allows clusters to have different sizes and shapes, allowing for more flexibility in capturing the true variance in the data.

<!-- Key Components of GMM -->
#### Key Components of GMM
There are three key components of GMM: Number of Gaussians Components, Expectation-Macimisatin (EM) Algorithm and Covariance Type. First set a Gaussian number (Components), using the EM algorithm to fit, iteratively improve the data possibilities of the given model. The covariance type of Gaussian determines the shape, size and orientation of the clusters.

<!-- The EM Algorithm in GMM -->
#### The EM Algorithm in GMM
The Expectation-Maximisation (EM) algorithm is a two-step process until convergence: Expectation Step (E-step) and Maximisation Step (M-step). The E-step is used to calculate the probability that each data point belongs to each cluster. And the M-step updates the parameters of the Gaussian function (mean, covariance, and mixing coefficients) to maximise the probability of the data given these distributions.

<!-- Advantages of GMM -->
#### Advantages of GMM
The GMM provides a probabilistic framework for soft clustering, which providing more information about uncertainty in data allocation. In addition, it can adapt to ellipsoidal cluster shapes, due to the flexible covariance structure.

<!-- Basic Code Implementation -->
#### Basic Code Implementation
Using sklearn, randomly create 100 data points, initialise a GMM model with three groups, and assign each data point. The data is visualised as a scatter plot, the clusters are represented by color-coded points, and the calculated cluster centers are marked with black dots. This demonstrates how GMM can group data using probability distributions.

```python
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import numpy as np

# Sample data
X = np.random.rand(100, 2)

# GMM model
gmm = GaussianMixture(n_components=3)
gmm.fit(X)
y_gmm = gmm.predict(X)

# Plotting
plt.scatter(X[:, 0], X[:, 1], c=y_gmm, cmap='viridis')
centers = gmm.means_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
plt.title('Gaussian Mixture Model')
plt.show()
```

<img src="https://github.com/Guoxuan-Li/GEOL0069_Week4/blob/main/Images/Gaussian.jpg?raw=true" width="500" height="400" />

<!-- Image Classification -->
### Image Classification
This section will explore the application of these unsupervised methods to image classification tasks, with a particular focus on distinguishing between sea ice and lead in Sentinel-2 images.

<!-- K-Means Implementation -->
#### K-Means Implementation
This script applies K-means clustering for Sentinel-2 satellite images to classify surface features. The clustering results are visualised and different colors are used to represent different clusters. This method can realise the unsupervised classification of Sentinel-2 images for environmental monitoring and research.

```python
import rasterio
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

base_path = "/content/drive/MyDrive/GEOL0069/2425/Week 4/Unsupervised Learning/S2A_MSIL1C_20190301T235611_N0207_R116_T01WCU_20190302T014622.SAFE/GRANULE/L1C_T01WCU_A019275_20190301T235610/IMG_DATA/" # You need to specify the path
bands_paths = {
    'B4': base_path + 'T01WCU_20190301T235611_B04.jp2',
    'B3': base_path + 'T01WCU_20190301T235611_B03.jp2',
    'B2': base_path + 'T01WCU_20190301T235611_B02.jp2'
}

# Read and stack the band images
band_data = []
for band in ['B4']:
    with rasterio.open(bands_paths[band]) as src:
        band_data.append(src.read(1))

# Stack bands and create a mask for valid data (non-zero values in all bands)
band_stack = np.dstack(band_data)
valid_data_mask = np.all(band_stack > 0, axis=2)

# Reshape for K-means, only including valid data
X = band_stack[valid_data_mask].reshape((-1, 1))

# K-means clustering
kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
labels = kmeans.labels_

# Create an empty array for the result, filled with a no-data value (e.g., -1)
labels_image = np.full(band_stack.shape[:2], -1, dtype=int)

# Place cluster labels in the locations corresponding to valid data
labels_image[valid_data_mask] = labels

# Plotting the result
plt.imshow(labels_image, cmap='viridis')
plt.title('K-means clustering on Sentinel-2 Bands')
plt.colorbar(label='Cluster Label')
plt.show()

del kmeans, labels, band_data, band_stack, valid_data_mask, X, labels_image
```

<img src="https://github.com/Guoxuan-Li/GEOL0069_Week4/blob/main/Images/K_mean%20clustering%20on%20s2%20bands.jpg?raw=true" width="500" height="400" />

This graph shows the K-means clustering results for Sentinel-2. Yellow areas may represent sea ice, while green areas indicate other surface types. Purple areas are the no data area.

<!-- GMM Implementation -->
#### GMM Implementation
The script applies GMM clustering to Sentinel-2 satellite images to classify surface features. The clustering results are visualised and different colors are used to represent different clusters.

```python
import rasterio
import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

# Paths to the band images
base_path = "/content/drive/MyDrive/GEOL0069/2425/Week 4/Unsupervised Learning/S2A_MSIL1C_20190301T235611_N0207_R116_T01WCU_20190302T014622.SAFE/GRANULE/L1C_T01WCU_A019275_20190301T235610/IMG_DATA/" # You need to specify the path
bands_paths = {
    'B4': base_path + 'T01WCU_20190301T235611_B04.jp2',
    'B3': base_path + 'T01WCU_20190301T235611_B03.jp2',
    'B2': base_path + 'T01WCU_20190301T235611_B02.jp2'
}

# Read and stack the band images
band_data = []
for band in ['B4']:
    with rasterio.open(bands_paths[band]) as src:
        band_data.append(src.read(1))

# Stack bands and create a mask for valid data (non-zero values in all bands)
band_stack = np.dstack(band_data)
valid_data_mask = np.all(band_stack > 0, axis=2)

# Reshape for GMM, only including valid data
X = band_stack[valid_data_mask].reshape((-1, 1))

# GMM clustering
gmm = GaussianMixture(n_components=2, random_state=0).fit(X)
labels = gmm.predict(X)

# Create an empty array for the result, filled with a no-data value (e.g., -1)
labels_image = np.full(band_stack.shape[:2], -1, dtype=int)

# Place GMM labels in the locations corresponding to valid data
labels_image[valid_data_mask] = labels

# Plotting the result
plt.imshow(labels_image, cmap='viridis')
plt.title('GMM clustering on Sentinel-2 Bands')
plt.colorbar(label='Cluster Label')
plt.show()
```
<img src="https://github.com/Guoxuan-Li/GEOL0069_Week4/blob/main/Images/GMM%20clustering%20on%20s2%20bands.jpg?raw=true" width="500" height="400" />

This graph shows the GMM clustering results for Sentinel-2. Yellow areas may represent sea ice, while green areas indicate other surface types. Purple areas are the no data area.

<!-- Altimetry Classification-->
### Altimetry Classification
This section applies unsupervised learning to classify Sentinel-3 altimetry data, focusing on distinguishing sea ice from leads using satellite-derived elevation measurements. This approach enhances the analysis of surface features, improving insights into ice dynamics and oceanographic processes.

<!-- Read in Functions Needed -->
#### Read in Functions Needed
Before delving into the modeling process, the data is pre-processed to ensure compatibility with the analytical model. Transform raw data into meaningful variables such as peaks and stack standard deviations (SSDs) and delete some NaN values from the dataset. Using netCDF4, numpy, matplotlib, scipy, glob and sklearn to extract and prrocess the Sentinel-3 SAR altimetry data to classify sea ice and leads. Visualize the data and plot the mean waveform and standard deviation of each class, the specific scripts are in the code folder.

```python
# mean and standard deviation for all echoes
mean_ice = np.mean(waves_cleaned[clusters_gmm==0],axis=0)
std_ice = np.std(waves_cleaned[clusters_gmm==0], axis=0)

plt.plot(mean_ice, label='ice')
plt.fill_between(range(len(mean_ice)), mean_ice - std_ice, mean_ice + std_ice, alpha=0.3)


mean_lead = np.mean(waves_cleaned[clusters_gmm==1],axis=0)
std_lead = np.std(waves_cleaned[clusters_gmm==1], axis=0)

plt.plot(mean_lead, label='lead')
plt.fill_between(range(len(mean_lead)), mean_lead - std_lead, mean_lead + std_lead, alpha=0.3)

plt.title('Plot of mean and standard deviation for each class')
plt.legend()
```
<img src="https://github.com/Guoxuan-Li/GEOL0069_Week4/blob/main/Images/plot%20of%20mean%20and%20sd%20for%20each%20class.jpg?raw=true" width="500" height="400" />


```python
x = np.stack([np.arange(1,waves_cleaned.shape[1]+1)]*waves_cleaned.shape[0])
plt.plot(x,waves_cleaned)  # plot of all the echos
plt.show()
```

<img src="https://github.com/Guoxuan-Li/GEOL0069_Week4/blob/main/Images/cleaned.jpg?raw=true" width="500" height="400" />


```python
# plot echos for the lead cluster
x = np.stack([np.arange(1,waves_cleaned[clusters_gmm==1].shape[1]+1)]*waves_cleaned[clusters_gmm==1].shape[0])
plt.plot(x,waves_cleaned[clusters_gmm==1])  # plot of all the echos
plt.show()
```
<img src="https://github.com/Guoxuan-Li/GEOL0069_Week4/blob/main/Images/echos%20for%20the%20lead%20cluster.jpg?raw=true" width="500" height="400" />


```python
# plot echos for the sea ice cluster
x = np.stack([np.arange(1,waves_cleaned[clusters_gmm==0].shape[1]+1)]*waves_cleaned[clusters_gmm==0].shape[0])
plt.plot(x,waves_cleaned[clusters_gmm==0])  # plot of all the echos
plt.show()
```
<img src="https://github.com/Guoxuan-Li/GEOL0069_Week4/blob/main/Images/echos%20for%20the%20sea%20ice%20cluster.jpg?raw=true" width="500" height="400" />


<!-- Scatter Plots of Clustered Data -->
### Scatter Plots of Clustered Data
Visualizes the clustering results using scatter plots, where different colors represent different clusters. The three graphs illustrate the relationship between sig_0, PP and SSD, where the first graph shows the relationship between sig_0 and PP, the second graph shows sig_0 and SSD, and the third graph shows PP and SSD. 

```python
plt.scatter(data_cleaned[:,0],data_cleaned[:,1],c=clusters_gmm)
plt.xlabel("sig_0")
plt.ylabel("PP")
plt.show()
plt.scatter(data_cleaned[:,0],data_cleaned[:,2],c=clusters_gmm)
plt.xlabel("sig_0")
plt.ylabel("SSD")
plt.show()
plt.scatter(data_cleaned[:,1],data_cleaned[:,2],c=clusters_gmm)
plt.xlabel("PP")
plt.ylabel("SSD")
```
<img src="https://github.com/Guoxuan-Li/GEOL0069_Week4/blob/main/Images/Scatter.png?raw=true" />


<!-- Waveform Alignment Using Cross-Correlation -->
### Waveform Alignment Using Cross-Correlation
This code aligns waveforms in the cluster where clusters_gmm == 0 by using cross-correlation. And visualise a subset of 10 equally spaced waveforms.

```python
from scipy.signal import correlate
 
# Find the reference point (e.g., the peak)
reference_point_index = np.argmax(np.mean(waves_cleaned[clusters_gmm==0], axis=0))
 
# Calculate cross-correlation with the reference point
aligned_waves = []
for wave in waves_cleaned[clusters_gmm==0][::len(waves_cleaned[clusters_gmm == 0]) // 10]:
    correlation = correlate(wave, waves_cleaned[clusters_gmm==0][0])
    shift = len(wave) - np.argmax(correlation)
    aligned_wave = np.roll(wave, shift)
    aligned_waves.append(aligned_wave)
 
# Plot aligned waves
for aligned_wave in aligned_waves:
    plt.plot(aligned_wave)
 
plt.title('Plot of 10 equally spaced functions where clusters_gmm = 0 (aligned)')
```
<img src="https://github.com/Guoxuan-Li/GEOL0069_Week4/blob/main/Images/gmm=0.jpg?raw=true" width="500" height="400" />

<!-- Compare with ESA data -->
### Compare with ESA data
In the ESA dataset, sea ice = 1 and lead = 2. Therefore, we need to subtract 1 from it so our predicted labels are comparable with the official product labels. Finally, creat a confusion matrix and classification report to summarise the accuracy of classifications.

<img src="https://github.com/Guoxuan-Li/GEOL0069_Week4/blob/main/Images/matrix.jpg?raw=true" width="500" height="400" />

The results showed high accuracy, which illustrate GMM can effectively distinguish sea ice and leads.







