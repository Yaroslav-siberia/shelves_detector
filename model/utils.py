import numpy
import os
from sklearn.neighbors import NearestNeighbors
import json
from json import JSONEncoder
import numpy as np
# computer vision
import cv2
# libs for efficientnet
import torch
import torchvision.transforms as transforms
import torchvision.transforms as T
import random
import gc

# Hyperparams
IMG_SIZE = 224  # use this param to resize cropped images
K_NEIGHBORS = 1  # k - neighbours
KNN_THRESHOLD = 0.27

device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")
PROJECT_PATH = os.path.abspath("")
PATH_TO_EMBEDDING_JSON = f"{PROJECT_PATH}/model/img_embeddings.json"
print(f'[utils]:Using {device} for inference')
print(f'[utils]:Project path: {PROJECT_PATH}')


model = torch.load('model/efficientnet_b0.pt')
model.eval()
print("[utils]: Model loaded to extract features")


transforms = T.Compose(
    [T.ToPILImage(), T.Resize((IMG_SIZE, IMG_SIZE)), T.ToTensor()])

def update_knn(PATH_TO_EMBEDDING_JSON):
    try:
        global knn ,categories, vectors
        categories, vectors = get_categories_and_embeddings(PATH_TO_EMBEDDING_JSON)
        knn = fit_knn(vectors)
        return True
    except Exception as e:
        print("[Error.refit_knn]", e)
        return False

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

def get_categories_and_embeddings(path_to_embedding_json: str):
    # читаю
    with open(path_to_embedding_json, "r") as read_file:
        all_categories = json.load(read_file)
    categories, embeds = [], []

    for category in all_categories:
        categories.append(list(category.values())[0])
        embeds.append(list(category.values())[1])
    return categories, embeds

def fit_knn(vectors: np.array):
    '''
        vectors - np.array of all embeddings in our json file
        '''
    knn = NearestNeighbors(metric='cosine', algorithm='brute', n_jobs=-1)
    knn.fit(vectors)
    return knn


def cluster_img(img: np.array):
    try:
        img_embed = vectorize_image(img)
        # test valid images
        label, score = k_nearest_label(
            categories, knn, K_NEIGHBORS, img_embed, show_score=True)
        if score < KNN_THRESHOLD:
            return 'None'
        else:
            return label
    except:
        cv2.imwrite(f'./{random.randint(0, 10)}.jpg',img)
        return 'None'



def create_embedding_json(path_to_embedding_json: str, dict_with_categories_embeddings: list):
    '''
    Функция для тестирования создания нового файла-словаря с векторными представлениями
    '''
    try:
        # Если такого файла нет, то создаю новый
        with open(path_to_embedding_json, "w", encoding='utf8') as write_file:
            json.dump(dict_with_categories_embeddings, write_file,
                      cls=NumpyArrayEncoder)
        return True
    except Exception as e:
        print("[Error.create_embedding_json]", e)
        return False


def update_embedding_json(path_to_embedding_json: str, dict_with_categories_embeddings: list):
    '''Функция для дозаписи новых данных размеченных данных в словарь с векторными представлениями
    '''
    try:
        if os.path.isfile(path_to_embedding_json):
            # читаю
            with open(path_to_embedding_json, "r") as read_file:
                all_categories = json.load(read_file)

            for dict_with_category in dict_with_categories_embeddings:

                all_categories.append(dict_with_category)
            # записываю
            with open(path_to_embedding_json, "w", encoding='utf8') as write_file:
                json.dump(all_categories, write_file,
                          cls=NumpyArrayEncoder)
        else:
            # Если такого файла нет, то создаю новый
            with open(path_to_embedding_json, "w", encoding='utf8') as write_file:
                json.dump(dict_with_categories_embeddings, write_file,
                          cls=NumpyArrayEncoder)
        return True
    except Exception as e:
        print("[Error.update_embedding_json]", e)
        return False


def k_nearest_label(categories: list, knn: NearestNeighbors, k: int, test_img_emb: np.ndarray, show_score: bool):
    '''
        categories - list of all categories from our json dict

                knn - pretraines knn-algorithm on of data

                k - (default = 1) k - nearest neighbors

                test_img_emb - vector of test image 

        return - text class label
        '''
    # Получаем ембеддинг самого изображения
    vec = test_img_emb.reshape(1, -1)
    dist, indices = knn.kneighbors(vec, n_neighbors=k)
    similar_images = [(categories[indices[i][0]], dist[i])
                      for i in range(len(indices))]
    label = similar_images[0][0]

    if show_score:
        score = similar_images[0][1][0]
        return label, score
    return label


def vectorize_image(img_array: np.array):
    '''
    img_array - np.array
    return - image_embedding
    '''
    a = transforms(img_array)
    a = a.unsqueeze(0)
    a_features = model.extract_features(a.to(device))
    a_features = torch.nn.functional.adaptive_avg_pool2d(a_features, 1)
    if len(a_features.shape) == 4:
        a_features = a_features.squeeze(2).squeeze(2).squeeze(0)
        a_features /= a_features.norm(dim=-1, keepdim=True)

    return a_features.cpu().detach().numpy()


def get_image(image_path: str):

    if os.path.isfile(image_path):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    else:
        image = None
        print("[Error.get_image]: Img doesnt exist")
        return None

categories, vectors = get_categories_and_embeddings(PATH_TO_EMBEDDING_JSON)
knn = fit_knn(vectors)