# 3.3 Confirmação dos Arquivos Selecionados para o Processamento (Opcional)

# Função para listar arquivos no GCS, ignorando diretórios vazios
def listar_arquivos_gcs(bucket, prefix):
    """Lista arquivos de um bucket no GCS com um prefixo específico, ignorando diretórios."""
    blobs = bucket.list_blobs(prefix=prefix)
    arquivos = [blob.name for blob in blobs if not blob.name.endswith('/')]  # Ignorar diretórios
    return arquivos

# Função para filtrar as amostras de interesse, conforme a lista de padrões
def filtrar_amostras_por_padroes(amostras, padroes):
    """Filtra a lista de amostras, verificando quais correspondem aos padrões fornecidos."""
    amostras_selecionadas = []
    for padrao in padroes:
        for amostra in amostras:
            if glob.fnmatch.fnmatch(amostra, f'sudamerica/{country}/training_samples/{padrao}'):
                amostras_selecionadas.append(amostra)
    return amostras_selecionadas

# Função para verificar se algum arquivo será sobrescrito
def verificar_sobrescrita(arquivos_existentes, arquivos_para_processar):
    arquivos_sobrescritos = []
    for arquivo in arquivos_para_processar:
        if arquivo in arquivos_existentes:
            arquivos_sobrescritos.append(arquivo)
    return arquivos_sobrescritos

# Inicializando a predição de áreas queimadas
print(f"\nIniciando predições de área queimada da região {region} do país {country.capitalize()}.")

for satellite_data in satellite_years:
    satellite = satellite_data['satellite']
    years = satellite_data['years']
    print(f"Processando os anos {years} do satélite {satellite.upper()}.")

# Listar amostras reais no GCS
gcs_folder_samples = f'sudamerica/{country}/training_samples'
samples_files = listar_arquivos_gcs(bucket, gcs_folder_samples)

# Filtrar as amostras que correspondem aos padrões da lista "images_train_test"
amostras_selecionadas = filtrar_amostras_por_padroes(samples_files, images_train_test)

print("\nAs amostras selecionadas para este processamento são:")
if amostras_selecionadas:
    for sample in amostras_selecionadas:
        print(f" - {sample}")
else:
    print("Nenhuma amostra correspondente foi encontrada.")

# Listar mosaicos reais no GCS
gcs_folder_mosaic = f'sudamerica/{country}/mosaics_col1_cog'
mosaics_files = listar_arquivos_gcs(bucket, gcs_folder_mosaic)
print("\nOs mosaicos que serão classificados são:")
if mosaics_files:
    for mosaic in mosaics_files:
        print(f" - {mosaic}")
else:
    print("Nenhum mosaico encontrado.")

# Listar arquivos já processados no GCS
gcs_folder_results = f'sudamerica/{country}/result_classified/'
resultados_existentes = listar_arquivos_gcs(bucket, gcs_folder_results)

# Gerar nomes dos arquivos que serão processados
arquivos_para_processar = []
for satellite_data in satellite_years:
    satellite = satellite_data['satellite']
    for year in satellite_data['years']:
        arquivo_nome = f"{gcs_folder_results}burned_area_{country}_{satellite}_v{version}_region{region}_{year}.tif"
        arquivos_para_processar.append(arquivo_nome)

# Verificar se há arquivos que serão sobrescritos
arquivos_sobrescritos = verificar_sobrescrita(resultados_existentes, arquivos_para_processar)

# Exibir resultados e alertas
print(f"\nOs resultados já existentes no bucket são:")
if resultados_existentes:
    for resultado in resultados_existentes:
        print(f" - {resultado}")
else:
    print("Nenhum resultado encontrado.")

if arquivos_sobrescritos:
    print("\n*** ALERTA: Os seguintes arquivos serão sobrescritos, e as versões antigas serão perdidas: ***")
    for arquivo in arquivos_sobrescritos:
        print(f" - {arquivo}")
else:
    print("\nNenhum arquivo será sobrescrito.")

print("\nOs novos arquivos processados serão armazenados neste endereço do bucket:")
print(f" - gs://{bucket_name}/{gcs_folder_results}")
