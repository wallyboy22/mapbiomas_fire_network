import os

# Instalar bibliotecas com pip
os.system('pip install rasterio')
os.system('pip install gcsfs ipywidgets')

# Instalar pacotes com apt-get
os.system('apt-get install -y libgdal-dev')
os.system('apt-get install -y gdal-bin')
os.system('apt-get install -y python3-gdal')
