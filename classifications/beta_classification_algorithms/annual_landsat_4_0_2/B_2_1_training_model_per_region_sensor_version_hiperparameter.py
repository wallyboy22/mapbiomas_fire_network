import os
import glob
import math
import time
import datetime
import numpy as np
import tensorflow as tf
from osgeo import gdal
from google.cloud import storage

os.environ['PROJ_LIB'] = '/usr/share/proj'
os.environ['GDAL_DATA'] = '/usr/share/gdal/2.2'
os.environ['GDAL_LIBRARY_PATH'] = '/usr/lib/libgdal.so'

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

def upload_to_gcs(local_file_path, bucket_name, gcs_destination_path):
    """
    Função para fazer upload de um arquivo para o Google Cloud Storage.

    Args:
        local_file_path (str): Caminho local do arquivo a ser enviado.
        bucket_name (str): Nome do bucket no Google Cloud Storage.
        gcs_destination_path (str): Caminho de destino no GCS.

    Returns:
        None
    """
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)
    blob = bucket.blob(gcs_destination_path)
    
    blob.upload_from_filename(local_file_path)
    print(f"[INFO] Arquivo {local_file_path} enviado para {gcs_destination_path} no bucket {bucket_name}.")


def log_message(message, log_file="treinamento_log.txt"):
    """
    Função para registrar uma mensagem no arquivo de log com timestamp.

    Args:
        message (str): Mensagem a ser registrada.
        log_file (str): Caminho do arquivo de log.

    Returns:
        None
    """
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(log_file, "a") as log:
        log.write(f"{timestamp} - {message}\n")

    print(message)  # Também imprime a mensagem no console


def treinar_modelo(versao, satelite, regiao, ano, bucket_name, country, hiperparametros=None, simulacao=False):
    """
    Função de treinamento com uma flag para simulação. Se simulacao=True, apenas emite prints simulados.
    """
    if simulacao:
        print(f"[SIMULACAO] Iniciando simulação do treinamento para versão: {versao}, satélite: {satelite}, região: {regiao}")
        return

    # Continua com o treinamento real se não for simulação
    log_message(f"Iniciando o treinamento para versão: {versao}, satélite: {satelite}, região: {regiao}")
    """
    Função de treinamento que aceita parâmetros de versão, satélite, região e hiperparâmetros.
    
    Parâmetros:
    - versao (str): Versão dos dados (ex. v1, v2).
    - satelite (str): Satélite dos dados (ex. l89, l78).
    - regiao (str): Região de processamento (ex. amazon, pacifico).
    - ano (int): Ano dos dados.
    - bucket_name (str): Nome do bucket no Google Cloud Storage.
    - country (str): Nome do país.
    - hiperparametros (dict): Dicionário de hiperparâmetros, se disponível.
    """

    folder_samples = f"./samples/{versao}_{satelite}_{regiao}"  # Define a pasta de amostras conforme os parâmetros
    folder_model = f"./models/{versao}_{satelite}_{regiao}"  # Pasta para salvar os modelos

    # Lista de imagens a serem processadas, conforme seleção do usuário
    images_train_test = [f"samples_fire_{versao}_{satelite}_{regiao}_*.tif"]

    # Criar pastas se não existirem
    if not os.path.exists(folder_samples):
        os.makedirs(folder_samples)
    if not os.path.exists(folder_model):
        os.makedirs(folder_model)

    # Hiperparâmetros padrão (caso não sejam fornecidos)
    default_hiperparametros = {
        'lr': 0.001,         # Taxa de aprendizado
        'BATCH_SIZE': 1000,  # Tamanho do lote
        'N_ITER': 7000,      # Número de iterações de treinamento
        'NUM_N_L1': 7,       # Neurônios na camada oculta 1
        'NUM_N_L2': 14,      # Neurônios na camada oculta 2
        'NUM_N_L3': 7,       # Neurônios na camada oculta 3
        'NUM_N_L4': 14,      # Neurônios na camada oculta 4
        'NUM_N_L5': 7,       # Neurônios na camada oculta 5
        'NUM_CLASSES': 2     # Número de classes (fogo, sem fogo)
    }

    # Usar hiperparâmetros fornecidos ou aplicar os padrões
    if hiperparametros:
        for param in default_hiperparametros:
            if param in hiperparametros:
                default_hiperparametros[param] = hiperparametros[param]

    # Agora `default_hiperparametros` contém os valores ajustados ou os padrões

    log_message(f"Hiperparâmetros utilizados: {default_hiperparametros}")

    # Continuar com o código de processamento e treinamento usando os hiperparâmetros ajustados
    lr = default_hiperparametros['lr']
    BATCH_SIZE = default_hiperparametros['BATCH_SIZE']
    N_ITER = default_hiperparametros['N_ITER']
    NUM_N_L1 = default_hiperparametros['NUM_N_L1']
    NUM_N_L2 = default_hiperparametros['NUM_N_L2']
    NUM_N_L3 = default_hiperparametros['NUM_N_L3']
    NUM_N_L4 = default_hiperparametros['NUM_N_L4']
    NUM_N_L5 = default_hiperparametros['NUM_N_L5']
    NUM_CLASSES = default_hiperparametros['NUM_CLASSES']

    # 5.2 Definição da Rede Neural Usando TensorFlow ###
    # A rede utiliza várias camadas totalmente conectadas (fully connected), normalização de dados,
    # e a função de perda de entropia cruzada para classificação. O otimizador Adam ajusta os pesos
    # e a acurácia é calculada durante o treinamento.

    # Função para criar uma camada totalmente conectada (fully connected)
    def fully_connected_layer(input, n_neurons, activation=None):
        """
        Cria uma camada totalmente conectada.

        :param input: Tensor de entrada da camada anterior
        :param n_neurons: Número de neurônios nesta camada
        :param activation: Função de ativação ('relu' ou None)
        :return: Saída da camada com ou sem ativação aplicada
        """
        input_size = input.get_shape().as_list()[1]  # Obtém o tamanho da entrada (número de features)

        # Inicializa pesos (W) com uma distribuição normal truncada e inicializa os bias (b) com zeros
        W = tf.Variable(tf.truncated_normal([input_size, n_neurons], stddev=1.0 / math.sqrt(float(input_size))), name='W')
        b = tf.Variable(tf.zeros([n_neurons]), name='b')

        # Aplica a transformação linear (Wx + b)
        layer = tf.matmul(input, W) + b

        # Aplica a função de ativação, se especificada
        if activation == 'relu':
            layer = tf.nn.relu(layer)

        return layer

    # Criação de um novo grafo computacional do TensorFlow
    graph = tf.Graph()
    with graph.as_default():  # Define o grafo como o padrão para operações

        # Definir placeholders para os dados de entrada e rótulos (labels)
        x_input = tf.placeholder(tf.float32, shape=[None, NUM_INPUT], name='x_input')  # Placeholder para os dados de entrada
        y_input = tf.placeholder(tf.int64, shape=[None], name='y_input')  # Placeholder para os rótulos (classe)

        # Normalizar os dados de entrada usando a média e o desvio padrão calculados anteriormente
        normalized = (x_input - data_mean) / data_std

        # Construir a rede neural com várias camadas totalmente conectadas
        hidden1 = fully_connected_layer(normalized, n_neurons=NUM_N_L1, activation='relu')
        hidden2 = fully_connected_layer(hidden1, n_neurons=NUM_N_L2, activation='relu')
        hidden3 = fully_connected_layer(hidden2, n_neurons=NUM_N_L3, activation='relu')
        hidden4 = fully_connected_layer(hidden3, n_neurons=NUM_N_L4, activation='relu')
        hidden5 = fully_connected_layer(hidden4, n_neurons=NUM_N_L5, activation='relu')

        """Camadas ocultas adicionais podem ser adicionadas aqui se necessário"""

        # Camada de saída final para produzir os logits (valores brutos para cada classe)
        logits = fully_connected_layer(hidden5, n_neurons=NUM_CLASSES)

        # Definir a função de perda: entropia cruzada softmax (para classificação multiclasse)
        cross_entropy = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y_input),
            name='cross_entropy_loss'
        )

        # Definir o otimizador: Adam com a taxa de aprendizado especificada
        optimizer = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

        # Operação para obter a classe prevista (classe com o maior logit)
        outputs = tf.argmax(logits, 1, name='predicted_class')

        # Métrica de acurácia: proporção de previsões corretas
        correct_prediction = tf.equal(outputs, y_input)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

        # Inicializar todas as variáveis do grafo
        init = tf.global_variables_initializer()

        # Definir o saver para salvar o estado do modelo durante o treinamento
        saver = tf.train.Saver()


    # 5.3 - Treinamento do modelo
    # --------------------------------------------------

    # Registrar o tempo de início do treinamento
    start_time = time.time()

    # Configurar opções de GPU para limitar o uso de memória (opcional)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)

    # Iniciar sessão TensorFlow para executar o gráfico
    log_message('[INFO] Iniciando sessão de treinamento com uso de GPU limitado a 33.3% da memória disponível...')
    with tf.Session(graph=graph, config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        sess.run(init)  # Inicializar todas as variáveis
        log_message('[INFO] Variáveis iniciais carregadas e sessão iniciada.')

        # Dicionário de dados de validação
        validation_dict = {
            x_input: validation_data[:, bi],
            y_input: validation_data[:, li]
        }

        log_message(f'[INFO] Iniciando o loop de treinamento com {N_ITER} iterações...')

        # Loop de treinamento: iterar sobre o número de iterações especificado
        for i in range(N_ITER + 1):
            # Selecionar um lote aleatório de dados de treinamento
            batch = training_data[np.random.choice(training_size, BATCH_SIZE, False), :]

            # Criar o dicionário de entrada para este lote
            feed_dict = {
              x_input: batch[:, bi],
              y_input: batch[:, li]
            }

            # Executar uma etapa do otimizador (etapa de treinamento)
            optimizer.run(feed_dict=feed_dict)

            # A cada 100 iterações, avaliar a acurácia e salvar o modelo
            if i % 100 == 0:
                # Calcular acurácia de validação
                acc = accuracy.eval(validation_dict) * 100

                # Salvar o checkpoint do modelo
                model_path = f'{folder_model}/col1_{country}_v{version}_r{region}_rnn_lstm_ckpt'
                saver.save(sess, model_path)

                # Mensagens de progresso e resultados intermediários
                log_message(f'[PROGRESSO] Iteração {i}/{N_ITER} - Acurácia de Validação: {acc:.2f}%')
                log_message(f'[INFO] Modelo salvo no caminho: {model_path}')

        # Tempo total gasto no treinamento
        end_time = time.time()
        training_time = end_time - start_time

 
        # Exibir o tempo total de treinamento em horas, minutos e segundos
        log_message(f'[INFO] Tempo total de treinamento: {time.strftime("%H:%M:%S", time.gmtime(training_time))}')

        # Fazer upload do modelo treinado para o Google Cloud Storage
        gcs_model_path = f"models/{versao}_{satelite}_{regiao}/model.ckpt"
        upload_to_gcs(model_path, bucket_name, gcs_model_path)

        log_message(f'[INFO] Modelo final salvo em {gcs_model_path} no Google Cloud Storage.')

    # Remover os arquivos de amostra após o término do treinamento
    log_message('[INFO] Removendo arquivos de amostra temporários...')
    remove_status = os.system(f'rm -rf {folder_samples}/samples_*')

    if remove_status == 0:
        log_message('[INFO] Arquivos de amostra removidos com sucesso.')
    else:
        log_message('[ERRO] Falha ao remover arquivos de amostra temporários.')