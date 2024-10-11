# 1. Instalação das bibliotecas necessárias

# Nesta subetapa, instalamos pacotes essenciais para processamento de imagens e integração com Google Earth Engine e Google Cloud Storage.

!pip install rasterio  # Library for working with geospatial raster data
!pip install gcsfs ipywidgets  # Instalação de bibliotecas para acesso ao GCS e widgets interativos

# Preparando o ambiente para o GDAL (Geospatial Data Abstraction Library)
!apt-get install -y libgdal-dev
!apt-get install -y gdal-bin
!apt-get install python3-gdal
