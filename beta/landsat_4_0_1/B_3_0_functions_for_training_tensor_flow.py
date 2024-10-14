# 4.1 Funções para rodar o treinamento

# Função para carregar uma imagem usando GDAL
def load_image(image_path):
    """
    Carrega uma imagem com GDAL no modo somente leitura.

    Args:
        image_path (str): Caminho para o arquivo da imagem.

    Returns:
        dataset (gdal.Dataset): Objeto GDAL contendo os dados da imagem.
    """
    dataset = gdal.Open(image_path, gdal.GA_ReadOnly)
    if dataset is None:
        raise FileNotFoundError(f"Erro ao carregar a imagem: {image_path}. Verifique o caminho.")
    return dataset

# Função para converter um dataset GDAL para um array NumPy
def convert_to_array(dataset):
    """
    Converte um dataset GDAL em um array NumPy.

    Args:
        dataset (gdal.Dataset): Objeto GDAL contendo os dados da imagem.

    Returns:
        np.ndarray: Array NumPy contendo os dados da imagem.
    """
    bands_data = [dataset.GetRasterBand(i + 1).ReadAsArray() for i in range(dataset.RasterCount)]
    return np.stack(bands_data, axis=2)  # Empilha as bandas ao longo do eixo Z
