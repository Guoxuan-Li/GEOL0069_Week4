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
Using NumPy, shapely, datatime, requests, pandas, xml.etree, os, json and folium, scripts fetch, process, and visualise data from the Copernicus Data Space Ecosystem. The specific codes are in the code folder.

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
![image](https://github.com/Guoxuan-Li/GEOL0069_Week4/blob/main/Images/1.png?raw=true)

