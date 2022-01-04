# Esse dataset possui imagens e anotações em XML sobre cada imagem. O Python tem algumas funções prontas para 
# extrair informações de um XML: 

from xml.etree import ElementTree
from os import listdir
from numpy import asarray
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
import matplotlib.pyplot as plt
from mrcnn.utils import extract_bboxes
from mrcnn.visualize import display_instances
from os import listdir
from numpy import zeros
from numpy import asarray
from numpy import expand_dims
from numpy import mean
from mrcnn.config import Config
from mrcnn.utils import Dataset
from mrcnn.utils import compute_ap
from mrcnn.model import load_image_gt
from mrcnn.model import mold_image

# Função para extrair os limites da caixa do arquivo de anotações XML:
def extrair_caixas(endereco_arquivo):
    arquivo = ElementTree.parse(endereco_arquivo) # carregando o arquivo e instanciando a função parse
    raiz = arquivo.getroot() # iniciando da raiz do documento
    # extraindo cada limite da caixa de dentro do marcador bndbox:
    caixas = []
    for caixa in raiz.findall('.//bndbox'):
        xmin = int(caixa.find('xmin').text)
        ymin = int(caixa.find('ymin').text)
        xmax = int(caixa.find('xmax').text)
        ymax = int(caixa.find('ymax').text)
        coordenadas = [xmin, ymin, xmax, ymax]
        caixas.append(coordenadas)
    # extraindo as dimensões da imagem de dentro do marcador size:
    largura = int(raiz.find('.//size/width').text)
    altura = int(raiz.find('.//size/height').text)
    return caixas, largura, altura


# Já existe uma função built-in load_mask(), mas como as máscaras podem ser escritas de diferentes formas, vamos
# sobreescrever a função built-in com uma função personalizada.

# Existem duas funções buit-in que iremos usar: add_class() e add_image(), que definem as classes e as imagens. 

# Na função add_class(), precisamos informar o nome do dataset, o número da classe (1 para canguru, 0 é reservado 
# para background), e o nome da classe (nesse caso, 1 é o número e 'kangaroo' é o nome).
# Resumindo: add_class('dataset_name', 1, 'kangaroo')

# Na função add_image(), precisamos informar: nome do dataset, nome do arquivo e o diretório que esse arquivo está. 
# Também podemos inserir outras informações nessa função, como o diretório que contém o arquivo com as anotações. 
# add_image('dataset_name', image_id='00001', path='kangaroo/images/00001.jpg', annotation='kangaroo/annots/00001.xml')

# O que queremos fazer é passar as informações de todas as imagens e classes. Vamos então criar uma função que vai
# varrer nossos diretórios e coletar essas informações de cada imagem, colocando cada uma dentro de add_image() e 
# add_class().
# Essa função será chamada de carrega_dataset(), e vamos aproveitar já para dividir o dataset em treino e teste:
def carrega_dataset(self, endereco_dataset, treino_selecionado=True):
    self.add_class("dataset", 1, "kangaroo") # aplica a função add_class()
    # organizando os diretórios onde estão as imagens e as anotações:
    endereco_imagens = endereco_dataset + '/images/' 
    endereco_anotacoes = endereco_dataset + '/annots/'
    # Encontrando todas as imagens e suas anotações para passar para a função add_image():
    for nome_imagem in listdir(endereco_imagens): # a função listdir retorna todos arquivos e diretórios existentes no caminho especificado      
        img_id = nome_imagem[:-4] # Pega o id da imagem (ignora o .jpg no final)
        if img_id in ['00090']: # Pula uma imagem com problema
            continue # quando o if é satisfeito, não continua, retornando para o próximo loop
        if treino_selecionado and int(img_id) >= 150: # Pula todas as imagens depois da 150 se estamos criando o dataset de treino
            continue
        if not treino_selecionado and int(img_id) < 150: # Pula todas as imagens antes da 150 se estamos criando o dataset de teste
            continue
        diretorio_completo_imagem = endereco_imagens + nome_imagem # Pega o diretório da imagem jpg
        diretorio_completo_anotacao = endereco_anotacoes + img_id + '.xml' # Pega o diretório da anotação XML da imagem
        self.add_image('dataset', image_id=img_id, path=diretorio_completo_imagem, annotation=diretorio_completo_anotacao) # Aplica na função add_image()


# Criando uma função que carrega a máscara de uma imagem específica:
def load_mask(self, im_id): # recebe o id da imagem como entrada
    informacoes_imagem = self.image_info[im_id] # recupera todas as informações passadas para add_image() a partir do image_id. É built-in.
    diretorio_anotacao = informacoes_imagem['annotation'] # salva o diretório da anotação dessa imagem
    caixas, l, a = self.extrair_caixas(diretorio_anotacao) # carrega o arquivo XML com base no diretório da anotação
    mascaras = zeros([a, l, len(caixas)], dtype='uint8') # Cria uma máscara de zeros com dimensão da imagem e profundidade da quantidade de caixas que há na imagem em questão. 
    # Colocando 1's em tod0 o tamanho de cada caixa dentro figura. Cada caixa será colocada em uma dimensão de profundidade:
    class_ids = []
    for i in range(len(caixas)): # varrendo uma caixa de cada vez
        caixa = caixas[i] # caixa atual
        x_inicio, x_final = caixa[1], caixa[3] # coleta x_min e x_máx
        y_inicio, y_final = caixa[0], caixa[2] # coleta y_min e y_máx
        mascaras[x_inicio:x_final, y_inicio:y_final, i] = 1 # colocando 1's em todo o retângulo formado
        class_ids.append(self.class_names.index('kangaroo')) # self.class_names contém o nome e o número das classes passadas para add_class()
    return mascaras, asarray(class_ids, dtype='int32') # retorna a máscara, e a classe (em um array int32. Se houver duas caixas, ambas das classes 1, essa saída será [1, 1])


# Implementando tudo que fizemos até aqui em um único código:
from os import listdir
from xml.etree import ElementTree
from numpy import zeros
from numpy import asarray
from mrcnn.utils import Dataset

class DatasetCanguru(Dataset):
    def carrega_dataset(self, endereco_dataset, treino_selecionado=True):
        self.add_class("dataset", 1, "kangaroo")
        endereco_imagens = endereco_dataset + '/images/'
        endereco_anotacoes = endereco_dataset + '/annots/'
        for nome_imagem in listdir(endereco_imagens):
            imgid = nome_imagem[:-4]
            if imgid in ['00090']:
                continue
            if treino_selecionado and int(imgid) >= 150:
                continue
            if not treino_selecionado and int(imgid) < 150:
                continue
            diretorio_completo_imagem = endereco_imagens + nome_imagem
            diretorio_completo_anotacao = endereco_anotacoes + imgid + '.xml'
            self.add_image('dataset', image_id=imgid, path=diretorio_completo_imagem, annotation=diretorio_completo_anotacao)

    def extrair_caixas(self, endereco_arquivo):
        arquivo = ElementTree.parse(endereco_arquivo)
        raiz = arquivo.getroot()
        caixas = []
        for caixa in raiz.findall('.//bndbox'):
            xmin = int(caixa.find('xmin').text)
            ymin = int(caixa.find('ymin').text)
            xmax = int(caixa.find('xmax').text)
            ymax = int(caixa.find('ymax').text)
            coordenadas = [xmin, ymin, xmax, ymax]
            caixas.append(coordenadas)
        largura = int(raiz.find('.//size/width').text)
        altura = int(raiz.find('.//size/height').text)
        return caixas, largura, altura

    def load_mask(self, id_imagem):
        informacoes_imagem = self.image_info[id_imagem]
        diretorio_anotacao = informacoes_imagem['annotation']
        caixas, l, a = self.extrair_caixas(diretorio_anotacao)
        mascaras = zeros([a, l, len(caixas)], dtype='uint8')
        id_classes = []
        for i in range(len(caixas)):
            caixa = caixas[i]
            x_inicio, x_final = caixa[1], caixa[3]
            y_inicio, y_final = caixa[0], caixa[2]
            mascaras[x_inicio:x_final, y_inicio:y_final, i] = 1
            id_classes.append(self.class_names.index('kangaroo'))
        return mascaras, asarray(id_classes, dtype='int32')


# Criando o dataset de treino:
dataset_treino = DatasetCanguru() # instancia a classe criada
dataset_treino.carrega_dataset('../dados/canguru/', treino_selecionado=True) # carrega o dataset
# usa a função built-in prepare() para preparar o dataset carregado (passa alguns atributos para variáveis internas)
dataset_treino.prepare()
print('Tamanho Treino: %d' % len(dataset_treino.image_ids))
 
# Criando o dataset de teste:
dataset_teste = DatasetCanguru()
dataset_teste.carrega_dataset('../dados/canguru/', treino_selecionado=False)
dataset_teste.prepare()
print('Tamanho Teste: %d' % len(dataset_teste.image_ids))

# Verificando se está tudo ok ao carregar uma imagem:


imagem_id = 3 # escolhendo uma imagem específica
imagem = dataset_treino.load_image(imagem_id) # aplica a função built-in load_image()
print(imagem.shape) # mostrando o shape da imagem
mascara, classes_ids = dataset_treino.load_mask(imagem_id) # aplica a função load_mask()
print(mascara.shape) # mostrando o shape da máscara da imagem
plt.imshow(imagem) # mostrando a imagem
plt.imshow(mascara[:, :, 0], cmap='gray', alpha=0.5) # mostrando a máscara da imagem. alpha é a transparência (1 é o default, menos que 1 fica mais transparente). cmap é o mapa de cores escolhido. Nesse caso, cinza.
plt.show()



# Mais uma verificação, dessa vez colocando o nome da classe em volta da imagem:


imagem_id = 3 # escolhendo uma imagem específica
imagem = dataset_treino.load_image(imagem_id) # aplica a função built-in load_image()
mascara, classes_ids = dataset_treino.load_mask(imagem_id) # aplica a função load_mask()
dimensoes_caixas = extract_bboxes(mascara) # aplica a função built-in extract_bboxes()
display_instances(imagem, dimensoes_caixas, mascara, classes_ids, dataset_treino.class_names) # mostra os dados coletados usando a função built-in display_instances()



# Criando uma classe de configurações básicas

from mrcnn.config import Config
from mrcnn.model import MaskRCNN

class ConfiguracoesCangurus(Config): 
    NAME = "configuracoes_cangurus" # atribui um nome para a configuração
    NUM_CLASSES = 2 # define o número de classes (background + kangaroo)
    STEPS_PER_EPOCH = 131 # define o número de passadas por epoch 
    GPU_COUNT = 1 # quantas GPUs serão utilizadas
    IMAGES_PER_GPU = 2 # quantas imagens serão passadas para a GPU de cada vez

config = ConfiguracoesCangurus()

# Criando o modelo:
modelo = MaskRCNN(mode='training', model_dir='../dados/canguru/treino', config=config)
# Em model_dir, informamos onde o modelo treinado será salvo. Em mode, podemos informar se é 'training' ou 'inference'
# Baixando o modelo COCO treinado: https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5
modelo.load_weights('../dados/canguru/mask_rcnn_coco.h5', by_name=True,
                    exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",  "mrcnn_bbox", "mrcnn_mask"])
# carregando pesos de um modelo já treinado, excluindo algumas camadas/layers inúteis para nosso caso
modelo.train(dataset_treino, dataset_teste, learning_rate=0.001, epochs=5, layers='heads')


# RODANDO O CÓDIGO COMPLETO

class DatasetCanguru(Dataset):
    def carrega_dataset(self, endereco_dataset, treino_selecionado=True):
        self.add_class("dataset", 1, "kangaroo")
        endereco_imagens = endereco_dataset + '/images/' 
        endereco_anotacoes = endereco_dataset + '/annots/'
        for nome_imagem in listdir(endereco_imagens): 
            imagem_id = nome_imagem[:-4] 
            if imagem_id in ['00090']:
                continue 
            if treino_selecionado and int(imagem_id) >= 150: 
                continue
            if not treino_selecionado and int(imagem_id) < 150: 
                continue
            diretorio_completo_imagem = endereco_imagens + nome_imagem 
            diretorio_completo_anotacao = endereco_anotacoes + imagem_id + '.xml' 
            self.add_image('dataset', image_id=imagem_id, path=diretorio_completo_imagem, annotation=diretorio_completo_anotacao) 
 
    def extrair_caixas(self, endereco_arquivo):
        arquivo = ElementTree.parse(endereco_arquivo) 
        raiz = arquivo.getroot() 
        caixas = []
        for caixa in raiz.findall('.//bndbox'):
            xmin = int(caixa.find('xmin').text)
            ymin = int(caixa.find('ymin').text)
            xmax = int(caixa.find('xmax').text)
            ymax = int(caixa.find('ymax').text)
            coordenadas = [xmin, ymin, xmax, ymax]
            caixas.append(coordenadas)
        largura = int(raiz.find('.//size/width').text)
        altura = int(raiz.find('.//size/height').text)
        return caixas, largura, altura
 
    def load_mask(self, imagem_id): 
        informacoes_imagem = self.image_info[imagem_id] 
        diretorio_anotacao = informacoes_imagem['annotation'] 
        caixas, l, a = self.extrair_caixas(diretorio_anotacao) 
        mascaras = zeros([a, l, len(caixas)], dtype='uint8') 
        classes_ids = []
        for i in range(len(caixas)): 
            caixa = caixas[i] 
            x_inicio, x_final = caixa[1], caixa[3] 
            y_inicio, y_final = caixa[0], caixa[2] 
            mascaras[x_inicio:x_final, y_inicio:y_final, i] = 1 
            classes_ids.append(self.class_names.index('kangaroo'))
        return mascaras, asarray(classes_ids, dtype='int32') 

# Quando formos chamar o modelo, tereremos que passar as configurações na forma de uma classe:
class ConfiguracoesCangurus(Config): 
    NAME = "configuracoes_cangurus" 
    NUM_CLASSES = 2 
    STEPS_PER_EPOCH = 131 
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2 
    
# Criando o dataset de treino:
dataset_treino = DatasetCanguru() 
dataset_treino.carrega_dataset('/home/natanael/Documents/DidaticaTech/kangaroo/', treino_selecionado=True) 
dataset_treino.prepare() 
print('Tamanho Treino: %d' % len(dataset_treino.image_ids))
 
# Criando o dataset de teste:
dataset_teste = DatasetCanguru()
dataset_teste.carrega_dataset('/home/natanael/Documents/DidaticaTech/kangaroo/', treino_selecionado=False)
dataset_teste.prepare()
print('Tamanho Teste: %d' % len(dataset_teste.image_ids))

# Preparando as configurações:
config = ConfiguracoesCangurus()

# Criando o modelo:
modelo = MaskRCNN(mode='training', model_dir='/home/natanael/Documents/DidaticaTech/kangaroo/treino', config=config) 
modelo.load_weights('/home/natanael/Documents/DidaticaTech/kangaroo/mask_rcnn_coco.h5', by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",  "mrcnn_bbox", "mrcnn_mask"]) 
modelo.train(dataset_treino, dataset_teste, learning_rate=0.001, epochs=5, layers='heads')

# O algoritmo treina com base nas caixas e máscaras. Provando:
# Quando se executa a função train() da classe MaskRCNN, ele pega para treinar o train_dataset, que é a classe que nós
# criamos: DatasetCanguru(). Essa classe contém a função load_mask(). Durante o treinamento, essa função será 
# chamada, pois ela está dentro de load_image_gt(), que por sua vez está dentro de data_generator(), que por sua
# vez está dentro de train(). Basta checar no código oficial: https://github.com/matterport/Mask_RCNN/blob/master/mrcnn/model.py

# Calculando a performance do modelo
# IoU (Intersection over Union) é a fórmula: (área sobreposta)/(área total). 
# Quando IoU > 0.5, é considerado que o modelo está bom. 
# Para verificar a performance do modelo, vamos calcular o IoU de cada imagem de teste e ver se está acima de 0.5.
# A taxa de sucessos (imagens com IoU > 0.5/ total_imagens) é chamada de Average Precision (AP).



# Essa primeira parte do código é igual ao que já utilizamos:
class DatasetCanguru(Dataset):
    def carrega_dataset(self, endereco_dataset, treino_selecionado=True):
        self.add_class("dataset", 1, "kangaroo")
        endereco_imagens = endereco_dataset + '/images/' 
        endereco_anotacoes = endereco_dataset + '/annots/'
        for nome_imagem in listdir(endereco_imagens): 
            imagem_id = nome_imagem[:-4] 
            if imagem_id in ['00090']:
                continue 
            if treino_selecionado and int(imagem_id) >= 150: 
                continue
            if not treino_selecionado and int(imagem_id) < 150: 
                continue
            diretorio_completo_imagem = endereco_imagens + nome_imagem 
            diretorio_completo_anotacao = endereco_anotacoes + imagem_id + '.xml' 
            self.add_image('dataset', image_id=imagem_id, path=diretorio_completo_imagem, annotation=diretorio_completo_anotacao) 
 
    def extrair_caixas(self, endereco_arquivo):
        arquivo = ElementTree.parse(endereco_arquivo) 
        raiz = arquivo.getroot() 
        caixas = []
        for caixa in raiz.findall('.//bndbox'):
            xmin = int(caixa.find('xmin').text)
            ymin = int(caixa.find('ymin').text)
            xmax = int(caixa.find('xmax').text)
            ymax = int(caixa.find('ymax').text)
            coordenadas = [xmin, ymin, xmax, ymax]
            caixas.append(coordenadas)
        largura = int(raiz.find('.//size/width').text)
        altura = int(raiz.find('.//size/height').text)
        return caixas, largura, altura
 
    def load_mask(self, imagem_id): 
        informacoes_imagem = self.image_info[imagem_id] 
        diretorio_anotacao = informacoes_imagem['annotation'] 
        caixas, l, a = self.extrair_caixas(diretorio_anotacao) 
        mascaras = zeros([a, l, len(caixas)], dtype='uint8') 
        classes_ids = []
        for i in range(len(caixas)): 
            caixa = caixas[i] 
            x_inicio, x_final = caixa[1], caixa[3] 
            y_inicio, y_final = caixa[0], caixa[2] 
            mascaras[x_inicio:x_final, y_inicio:y_final, i] = 1 
            classes_ids.append(self.class_names.index('kangaroo'))
        return mascaras, asarray(classes_ids, dtype='int32') 

# Criando uma classe para as configurações das previsões:
class ConfiguracoesPrevisoes(Config): 
    NAME = "configuracoes_previsoes"
    NUM_CLASSES = 2
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

# Criando uma função para calcular o AP:
def avalia_modelo(dataset, modelo, cfg):
    APs = []
    for imagem_id in dataset.image_ids:
        # Extraindo cada imagem, sua caixa e sua máscara:
        dim_imagem, dim_image_bruta, id_classe_gt, dim_caixas_gt, mascara_gt = load_image_gt(dataset, cfg, imagem_id, use_mini_mask=False) #mini_mask é um parâmetro pra diminuir o tamanho da máscara (caso existam máscaras muito grandes)
        # dim_imagem e dim_image_bruta são as dimensões atuais da imagem e suas dimensões antes de redimensionar e cortar
        # Agora vamos subtrair todos os pixels da imagem pelo valor do pixel médio. Isso se chama "centering",  
        # é uma prática comum no pré-processamento de imagens, pois coloca a distribuição dos valores dos pixels
        # centrada em zero (isso facilita a sensibilidade do aprendizado na CNN):
        imagem_centrada = mold_image(dim_imagem, cfg)
        # Quando estamos treinando o modelo, estamos passando as imagens em batches para um placeholder. Ou seja, 
        # a dimensão de entrada é não apenas a dimensão das imagens, mas também tem uma dimensão que informa a 
        # quantidade de imagens. No treino era assim: [batch_size, img_width, img_height, number_of_channels]
        # Na hora de fazer previsões, precisamos fazer o mesmo. Então se queremos passar uma imagem de cada vez,
        # precisamos mesmo assim criar uma dimensão que diga que há somente uma imagem nesse lote. 
        # Isso se consegue facilmente com a função expand_dims() do numpy. 
        # Exemplo: a imagem atual possui o formato [img_width, img_height, number_of_channels]
        # Usando expand_dims(image, axis=0), teremos: [1, img_width, img_height, number_of_channels]
        amostra = expand_dims(imagem_centrada, 0)
        # Fazendo as previsões:
        coleta_infos = modelo.detect(amostra, verbose=0) # a função detect() retorna as caixas, a classe, o score da classe e a máscara.
        # A saída da função detect() é uma lista de dicionários. Vamos coletar os dados do dicionário recém criado:
        dados = coleta_infos[0]
        # Usando a função built-in que calcula o AP:
        AP, _, _, _ = compute_ap(dim_caixas_gt, id_classe_gt, mascara_gt, dados["rois"], dados["class_ids"], dados["scores"], dados['masks'], iou_threshold=0.5)
        # Para a função acima, estamos passando uma imagem de cada vez, então podemos armazenar isso:
        APs.append(AP)
    # Agora que todos os valores foram armazenados, vamos calcular a média total:
    media_AP = mean(APs)
    return media_AP
 
# Criando o dataset de treino:
dataset_treino = DatasetCanguru() 
dataset_treino.carrega_dataset('/home/natanael/Documents/DidaticaTech/kangaroo/', treino_selecionado=True) 
dataset_treino.prepare() 
print('Tamanho Treino: %d' % len(dataset_treino.image_ids))
 
# Criando o dataset de teste:
dataset_teste = DatasetCanguru()
dataset_teste.carrega_dataset('/home/natanael/Documents/DidaticaTech/kangaroo/', treino_selecionado=False)
dataset_teste.prepare()
print('Tamanho Teste: %d' % len(dataset_teste.image_ids))

cfg = ConfiguracoesPrevisoes()

modelo = MaskRCNN(mode='inference', model_dir='/home/natanael/Documents/DidaticaTech/kangaroo/modelo_testando', config=cfg)
# Carregando os dados do modelo já treinado:
modelo.load_weights('/home/natanael/Documents/DidaticaTech/kangaroo/treino/configuracoes_cangurus20200130T1834/mask_rcnn_configuracoes_cangurus_0005.h5', by_name=True)
# Avaliando a performance nos dados de treino:
media_AP_treino = avalia_modelo(dataset_treino, modelo, cfg)
print("Média AP total dataset_treino: %.3f" % media_AP_treino)
# Avaliando a performance nos dados de teste:
media_AP_teste = avalia_modelo(dataset_teste, modelo, cfg)
print("Média AP total dataset_teste: %.3f" % media_AP_teste)


# In[8]:


# Mostrando a detecção de cangurus

from os import listdir
from xml.etree import ElementTree
from numpy import zeros
from numpy import asarray
from numpy import expand_dims
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from mrcnn.model import mold_image
from mrcnn.utils import Dataset
from matplotlib.pyplot import figure
 
# Essa primeira parte do código é igual ao que já utilizamos
class DatasetCanguru(Dataset):
    def carrega_dataset(self, endereco_dataset, treino_selecionado=True):
        self.add_class("dataset", 1, "kangaroo")
        endereco_imagens = endereco_dataset + '/images/' 
        endereco_anotacoes = endereco_dataset + '/annots/'
        for nome_imagem in listdir(endereco_imagens): 
            imagem_id = nome_imagem[:-4] 
            if imagem_id in ['00090']:
                continue 
            if treino_selecionado and int(imagem_id) >= 150: 
                continue
            if not treino_selecionado and int(imagem_id) < 150: 
                continue
            diretorio_completo_imagem = endereco_imagens + nome_imagem 
            diretorio_completo_anotacao = endereco_anotacoes + imagem_id + '.xml' 
            self.add_image('dataset', image_id=imagem_id, path=diretorio_completo_imagem, annotation=diretorio_completo_anotacao) 
 
    def extrair_caixas(self, endereco_arquivo):
        arquivo = ElementTree.parse(endereco_arquivo) 
        raiz = arquivo.getroot() 
        caixas = []
        for caixa in raiz.findall('.//bndbox'):
            xmin = int(caixa.find('xmin').text)
            ymin = int(caixa.find('ymin').text)
            xmax = int(caixa.find('xmax').text)
            ymax = int(caixa.find('ymax').text)
            coordenadas = [xmin, ymin, xmax, ymax]
            caixas.append(coordenadas)
        largura = int(raiz.find('.//size/width').text)
        altura = int(raiz.find('.//size/height').text)
        return caixas, largura, altura
 
    def load_mask(self, imagem_id): 
        informacoes_imagem = self.image_info[imagem_id] 
        diretorio_anotacao = informacoes_imagem['annotation'] 
        caixas, l, a = self.extrair_caixas(diretorio_anotacao) 
        mascaras = zeros([a, l, len(caixas)], dtype='uint8') 
        classes_ids = []
        for i in range(len(caixas)): 
            caixa = caixas[i] 
            x_inicio, x_final = caixa[1], caixa[3] 
            y_inicio, y_final = caixa[0], caixa[2] 
            mascaras[x_inicio:x_final, y_inicio:y_final, i] = 1 
            classes_ids.append(self.class_names.index('kangaroo'))
        return mascaras, asarray(classes_ids, dtype='int32') 

# Criando uma classe para as configurações das previsões:
class ConfiguracoesPrevisoes(Config): 
    NAME = "configuracoes_previsoes"
    NUM_CLASSES = 2
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

# Criando função que mostra n imagens de teste, cada uma com sua máscara real e com a caixa prevista pelo modelo:
def plotando_atual_vs_prevista(dataset, modelo, cfg, n_imagens=5):
    for i in range(n_imagens): # varrendo as primeiras 2 imagens do dataset de teste
        imagem = dataset.load_image(i) # carrega cada imagem
        mascara, _ = dataset.load_mask(i) # carrega cada máscara
        imagem_centrada = mold_image(imagem, cfg) # aplicando centering
        amostra = expand_dims(imagem_centrada, 0) # criando a dimensão extra do batch_size
        dados = modelo.detect(amostra, verbose=0)[0] # fazendo a previsão dessa imagem
        figura = plt.figure(figsize=(15,15)) # tamanho da figura total (terá que comportar 4 imagens nesse caso)
        
        # Criando primeiro os subplots para mostrar as imagens de teste com suas máscaras reais:
        figura.add_subplot(n_imagens, 2, i*2+1) # quebra as linhas da figura em 2 partes e as colunas em 2 partes. Para cada imagem i, plota ela nos quadrantes da coluna esquerda (quadrantes ímpares)
        plt.imshow(imagem) # mostra a imagem
        plt.title('Gabarito') 
        # plotando as máscaras reais:
        for j in range(mascara.shape[2]):
            plt.imshow(mascara[:, :, j], cmap='gray', alpha=0.3) # desenha a máscara
            
        # Criando os subplots para mostrar as imagens de teste com suas caixas previstas pelo modelo:
        figura.add_subplot(n_imagens, 2, i*2+2) # quebra as linhas da figura em 2 partes e as colunas em 2 partes. Para cada imagem i, plota ela nos quadrantes da coluna direita (quadrantes pares)
        plt.imshow(imagem) # mostra a imagem 
        plt.title('Previsão do modelo')
        eixos = plt.gca() # gca significa "get current axis". Ele seleciona os eixos da figura atual.
        # plotando as caixas previstas:
        for caixa in dados['rois']: # coleta as informações da caixa prevista
            y1, x1, y2, x2 = caixa # coloca cada coordenada da caixa em uma variável
            largura, altura = x2 - x1, y2 - y1 # calcula a espessura e a altura da caixa
            retangulo = Rectangle((x1, y1), largura, altura, fill=False, color='red') # cria um retângulo (caixa)
            eixos.add_patch(retangulo) # adiciona o retângulo criado na figura
    # Resumindo: para cada imagem de teste, mostramos a imagem com sua máscara na esquerda e a imagem com sua caixa
    # prevista pelo modelo na direita.            
    plt.show()

# Criando o dataset de treino:
dataset_treino = DatasetCanguru() 
dataset_treino.carrega_dataset('/home/natanael/Documents/DidaticaTech/kangaroo/', treino_selecionado=True) 
dataset_treino.prepare() 
print('Tamanho Treino: %d' % len(dataset_treino.image_ids))
 
# Criando o dataset de teste:
dataset_teste = DatasetCanguru()
dataset_teste.carrega_dataset('/home/natanael/Documents/DidaticaTech/kangaroo/', treino_selecionado=False)
dataset_teste.prepare()
print('Tamanho Teste: %d' % len(dataset_teste.image_ids))

cfg = ConfiguracoesPrevisoes()

modelo = MaskRCNN(mode='inference', model_dir='/home/natanael/Documents/DidaticaTech/kangaroo/modelo_testando', config=cfg)

# Escolhendo qual modelo treinado utilizar para fazer a detecção de cangurus:
modelo.load_weights('/home/natanael/Documents/DidaticaTech/kangaroo/treino/configuracoes_cangurus20200130T1834/mask_rcnn_configuracoes_cangurus_0002.h5', by_name=True)

# Mostrando as previsões no dataset de teste:
plotando_atual_vs_prevista(dataset_teste, modelo, cfg)


# In[9]:


# Mostrando a aplicação funcionando com uma imagem nova aleatória
from os import listdir
from xml.etree import ElementTree
from numpy import zeros
from numpy import asarray
from numpy import expand_dims
from matplotlib import pyplot
from matplotlib.patches import Rectangle
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from mrcnn.model import mold_image
from mrcnn.utils import Dataset
from matplotlib.pyplot import figure
from skimage.io import imread
 

class DatasetCanguru(Dataset):
    def carrega_dataset(self, endereco_dataset, treino_selecionado=True):
        self.add_class("dataset", 1, "kangaroo")
        endereco_imagens = endereco_dataset + '/images/' 
        endereco_anotacoes = endereco_dataset + '/annots/'
        for nome_imagem in listdir(endereco_imagens): 
            imagem_id = nome_imagem[:-4] 
            if imagem_id in ['00090']:
                continue 
            if treino_selecionado and int(imagem_id) >= 150: 
                continue
            if not treino_selecionado and int(imagem_id) < 150: 
                continue
            diretorio_completo_imagem = endereco_imagens + nome_imagem 
            diretorio_completo_anotacao = endereco_anotacoes + imagem_id + '.xml' 
            self.add_image('dataset', image_id=imagem_id, path=diretorio_completo_imagem, annotation=diretorio_completo_anotacao) 
 
    def extrair_caixas(self, endereco_arquivo):
        arquivo = ElementTree.parse(endereco_arquivo) 
        raiz = arquivo.getroot() 
        caixas = []
        for caixa in raiz.findall('.//bndbox'):
            xmin = int(caixa.find('xmin').text)
            ymin = int(caixa.find('ymin').text)
            xmax = int(caixa.find('xmax').text)
            ymax = int(caixa.find('ymax').text)
            coordenadas = [xmin, ymin, xmax, ymax]
            caixas.append(coordenadas)
        largura = int(raiz.find('.//size/width').text)
        altura = int(raiz.find('.//size/height').text)
        return caixas, largura, altura
 
    def load_mask(self, imagem_id): 
        informacoes_imagem = self.image_info[imagem_id] 
        diretorio_anotacao = informacoes_imagem['annotation'] 
        caixas, l, a = self.extrair_caixas(diretorio_anotacao) 
        mascaras = zeros([a, l, len(caixas)], dtype='uint8') 
        classes_ids = []
        for i in range(len(caixas)): 
            caixa = caixas[i] 
            x_inicio, x_final = caixa[1], caixa[3] 
            y_inicio, y_final = caixa[0], caixa[2] 
            mascaras[x_inicio:x_final, y_inicio:y_final, i] = 1 
            classes_ids.append(self.class_names.index('kangaroo'))
        return mascaras, asarray(classes_ids, dtype='int32') 

# Criando uma classe para as configurações das previsões:
class ConfiguracoesPrevisoes(Config): 
    NAME = "configuracoes_previsoes"
    NUM_CLASSES = 2
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


# Criando função que mostra n imagens de teste, cada uma com sua máscara real e com a caixa prevista pelo modelo:
def plotando_atual_vs_prevista(imagem, modelo, cfg):
    imagem_centrada = mold_image(imagem, cfg) # aplicando centering
    amostra = expand_dims(imagem_centrada, 0) # criando a dimensão extra do batch_size
    dados = modelo.detect(amostra, verbose=0)[0] # fazendo a previsão dessa imagem
    figura = plt.figure(figsize=(15,15)) # tamanho da figura total (irá comportar 2 imagens nesse caso)
    # Criando o subplot para mostrar a imagem atual sem previsão:
    figura.add_subplot(1, 2, 1) # quebra a figura em duas partes nas colunas. Irá colocar a imagem carregada na esquerda.
    plt.imshow(imagem) # mostra a imagem
    plt.title('Imagem original') 
    figura.add_subplot(1, 2, 2) # quebra a figura em duas partes nas colunas. Irá colocar a detecção do modelo na direita.
    plt.imshow(imagem) # mostra a imagem
    plt.title('Previsao')
    eixos = plt.gca()
    # plotando as caixas previstas:
    for caixa in dados['rois']:
        y1, x1, y2, x2 = caixa
        largura, altura = x2 - x1, y2 - y1
        retangulo = Rectangle((x1, y1), largura, altura, fill=False, color='red')
        eixos.add_patch(retangulo) # adiciona o retângulo previsto pelo modelo na figura
    plt.show()

# Carregando a imagem:
imagem_canguru = imread('../dados/canguru/imagens/imagem_canguru.jpg')

cfg = ConfiguracoesPrevisoes()

modelo = MaskRCNN(mode='inference', model_dir='../dados/canguru/modelo_testando', config=cfg)
modelo.load_weights('../dados/canguru/mask_rcnn_configuracoes_cangurus_0002.h5', by_name=True)

# Mostrando as previsões da imagem:
plotando_atual_vs_prevista(imagem_canguru, modelo, cfg)
