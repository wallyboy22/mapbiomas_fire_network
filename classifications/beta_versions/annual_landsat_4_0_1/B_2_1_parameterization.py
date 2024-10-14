# 3.2 Definição de Parâmetros para o Processamento

# Definição do país a ser processado
country = 'bolivia'  # Opções: ['bolivia', 'colombia', 'chile', 'peru', 'paraguay', 'guyana']

# Inicializa a API do Google Earth Engine para o projeto específico, substituindo '{country}' pelo país desejado
# ee.Initialize(project=f'mapbiomas-{country}')
# print('Google Earth Engine API initialized successfully')

# Definição de parâmetros de versão e região
version = 1
region = '2'  # Opções: ['1', '2', '3', '4', '5']

# Definir diretórios para o armazenamento de dados e saída do modelo
folder = f'/content/mapbiomas-fire/sudamerica/{country}'  # Diretório principal onde os dados são armazenados

folder_samples = f'{folder}/training_samples'  # Diretório para armazenamento de dados de amostra
folder_model = f'{folder}/models_col1'  # Diretório para armazenamento da saída dos modelos

# Garantir que o diretório folder_samples exista
if not os.path.exists(folder_samples):
    os.makedirs(folder_samples)

# Garantir que o diretório folder_model exista
if not os.path.exists(folder_model):
    os.makedirs(folder_model)

# Diretórios para armazenamento temporário de imagens e mosaicos COG
folder_images = f'{folder}/tmp1'  # Diretório para armazenamento temporário de imagens
folder_mosaic = f'{folder}/mosaics_cog'  # Diretório para arquivos COG (Cloud-Optimized GeoTIFF)
sulfix = ''  # Sufixo para arquivos de saída (se necessário)

# Lista de padrões de arquivos para imagens de treinamento e teste, baseada no país, versão e região selecionados
images_train_test = [
    f'samples_fire_v{version}_l89_{country}_r{region}*.tif',
    f'samples_fire_v{version}_l78_{country}_r{region}*.tif'
]

# Definir pares de satélites e anos para o processamento
satellite_years = [
    {'satellite': 'l78', 'years': [2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021]},
    # {'satellite': 'l78', 'years': [2018, 2019, 2020, 2021]},
    {'satellite': 'l89', 'years': [2022, 2023]}
]
