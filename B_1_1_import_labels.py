B_1_1 Instalação de bibliotecas necessárias para processamento de imagens e modelagem.
# Importar bibliotecas necessárias
import os  # Library for interacting with the operating system
print('Successfully imported os')

# Instalar e importar Earth Engine API
#!pip install earthengine-api --upgrade  # Uncomment if you need to install or upgrade Earth Engine API
import ee  # Google Earth Engine Python API
print('Successfully imported ee')

from google.cloud import storage  # Google Cloud Storage client library
print('Successfully imported google.cloud.storage')

from IPython import display  # Display utilities for Jupyter notebooks
print('Successfully imported IPython display')

import math  # Mathematical functions
print('Successfully imported math')

from matplotlib import pyplot as plt  # Visualization library for plots and images
print('Successfully imported matplotlib.pyplot')

import numpy as np  # Library for numerical operations on arrays and matrices
print('Successfully imported numpy')

import pandas as pd  # Data analysis and manipulation library
print('Successfully imported pandas')

import seaborn as sns  # Data visualization library based on matplotlib
print('Successfully imported seaborn')

from scipy import ndimage  # Multidimensional image processing
print('Successfully imported scipy.ndimage')

import rasterio  # Reading and writing geospatial raster data
print('Successfully imported rasterio')

from rasterio.mask import mask  # Function to mask raster data with shapes
print('Successfully imported rasterio.mask')

import tempfile  # Temporary file creation utilities
print('Successfully imported tempfile')

import urllib.request as urllib  # For downloading files from the internet
print('Successfully imported urllib.request')

import zipfile  # Tools to handle ZIP archives
print('Successfully imported zipfile')

import datetime  # Handling date and time
print('Successfully imported datetime')

import time  # Time-related functions
print('Successfully imported time')

import string  # String handling utilities
print('Successfully imported string')

import glob  # File pattern matching
print('Successfully imported glob')

from osgeo import gdal  # Geospatial Data Abstraction Library for raster data processing
print('Successfully imported osgeo.gdal')

from shapely.geometry import shape, mapping, box  # Geometric objects and operations
print('Successfully imported shapely.geometry')

from shapely.ops import transform  # Geometric transformations
print('Successfully imported shapely.ops')

import pyproj  # Coordinate reference system transformations
print('Successfully imported pyproj')

# import tensorflow as tf  # TensorFlow for building neural networks (version 2.x)
import tensorflow.compat.v1 as tf  # TensorFlow compatibility mode for version 1.x
tf.disable_v2_behavior()  # Disable TensorFlow 2.x behaviors and enable 1.x style
print('Successfully imported tensorflow.compat.v1')

import gcsfs  # Pythonic file-system interface to Google Cloud Storage
print('Successfully imported gcsfs')

import ipywidgets as widgets  # Library for creating interactive widgets in Jupyter notebooks
print('Successfully imported ipywidgets')

from IPython.display import display, HTML, clear_output  # Utilities for displaying content in Jupyter notebooks
print('Successfully imported IPython display functions')

from ipywidgets import VBox, HBox  # Containers for arranging widgets in vertical or horizontal boxes
print('Successfully imported ipywidgets VBox and HBox')

from tqdm import tqdm  # Progress bar for long-running operations
print('Successfully imported tqdm')

import time  # Time-related functions
print('Successfully imported time')

os.environ['PROJ_LIB'] = '/usr/share/proj'
os.environ['GDAL_DATA'] = '/usr/share/gdal/2.2'
os.environ['GDAL_LIBRARY_PATH'] = '/usr/lib/libgdal.so'