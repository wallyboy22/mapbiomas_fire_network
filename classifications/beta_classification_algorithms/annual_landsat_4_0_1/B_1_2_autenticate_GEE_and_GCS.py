# 2. Autenticação com Google Earth Engine e Google Cloud Storage

# Autenticação com o Google Earth Engine (caso necessário)
ee.Authenticate()

# Inicializar o Google Earth Engine com o projeto definido
country = 'bolivia'
ee_project = f'mapbiomas-{country}'  # Defina o nome do projeto corretamente
ee.Initialize(project=ee_project)
print('Google Earth Engine API initialized successfully')

# Inicializar o cliente do Google Cloud Storage e definir o nome do bucket
bucket_name = 'mapbiomas-fire'
client = storage.Client()
bucket = client.get_bucket(bucket_name)
print(f'Connected to Google Cloud Storage bucket: {bucket_name}')

# Autenticação com Google Cloud (necessária no Colab)
from google.colab import auth
auth.authenticate_user()
print('Google Cloud authentication successful')
