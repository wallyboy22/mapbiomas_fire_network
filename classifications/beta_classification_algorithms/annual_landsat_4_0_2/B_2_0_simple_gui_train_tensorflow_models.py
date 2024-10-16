import gcsfs
import ipywidgets as widgets
from IPython.display import display, clear_output

# Inicializa o sistema de arquivos do Google Cloud Storage
fs = gcsfs.GCSFileSystem(project=bucket_name)

# Define o caminho base para as pastas dos países
pasta_base = 'mapbiomas-fire/sudamerica/'

# Função para listar os países (pastas principais)
def listar_paises(pasta_base):
    pastas = fs.ls(pasta_base)
    paises = [pasta.split('/')[-1] for pasta in pastas if pasta.split('/')[-1]]  # Remove itens vazios
    return paises

# Função para listar o conteúdo da subpasta "training_samples" de cada país
def listar_training_samples(pasta_pais):
    pasta_training = f"{pasta_base}{pasta_pais}/training_samples/"
    try:
        arquivos = fs.ls(pasta_training)
        return [arquivo.split('/')[-1] for arquivo in arquivos if arquivo.split('/')[-1]]  # Remove itens vazios
    except FileNotFoundError:
        return []  # Retorna uma lista vazia se a subpasta não existir

# Função para formatar os arquivos de forma personalizada, semelhante ao exemplo JS
def formatar_arquivos(arquivos):
    formatted_list = []
    
    for arquivo in arquivos:
        split = arquivo.split('_')  # Divide o nome do arquivo em partes
        if len(split) >= 6:  # Garantir que há partes suficientes para a formatação
            formatted = f'trainings_{split[2]}_{split[3]}_{split[4]}_{split[5]}'  # Formatação personalizada
            if formatted not in formatted_list:
                formatted_list.append(formatted)  # Adicionar se ainda não estiver na lista
    
    return formatted_list

# Função para exibir o conteúdo de "training_samples" ao selecionar um país
def ao_selecionar_pais(change):
    clear_output(wait=True)  # Limpar a saída anterior, mas manter o dropdown
    display(dropdown_paises)  # Reexibir o dropdown para mantê-lo visível
    
    pais_selecionado = change['new']
    print(f"País selecionado: {pais_selecionado}")
    
    # Listar e exibir os arquivos na pasta "training_samples"
    arquivos_training = listar_training_samples(pais_selecionado)
    if arquivos_training:
        print(f"Arquivos na pasta 'training_samples':")
        for arquivo in arquivos_training:
            print(f'  - {arquivo}')
        
        # Formatar os arquivos
        arquivos_formatados = formatar_arquivos(arquivos_training)
        
        print("\nArquivos formatados:")
        for arquivo in arquivos_formatados:
            print(f'  - {arquivo}')
    else:
        print("Nenhum arquivo encontrado na pasta 'training_samples'.")

# Widget de dropdown para selecionar o país
dropdown_paises = widgets.Dropdown(
    options=listar_paises(pasta_base),
    description='Países:',
    disabled=False
)

# Exibir o dropdown inicialmente
display(dropdown_paises)

# Vincular o evento de mudança de valor ao dropdown
dropdown_paises.observe(ao_selecionar_pais, names='value')
