# Image embedding json

> Файл хранит метку и векторное представление о каждом размеченном объекте

*  JSON имеет структуру: [{"name": str , "embedding" np.array },]

# UTILS.py

Гиперпараметры: 
* KNN_THRESHOLD - порог для нахождения близжайшего
* IMG_SIZE - размер картинки для резайза
* K_NEIGHBORS - количество соседей

**Методы:**

**def cluster_img(img : np.array)**

> Функция принимает на вход картинку и по порогу KNN_THRESHOLD относит к определенному кластеру
* Вход: img : np.array - картинка
* Выход: label: str | 'None' 



**def vectorize_image(img : np.array)**
> Функция принимает на вход картинку и возвращает векторное предствление
* Вход: img : np.array - картинка
* Выход: img_embedding: np.array


**def create_embedding_json(path_to_embedding_json : str, dict_with_categories_embeddings: list)**
> Функция создания нового Image embedding json (файла-словаря) с векторными представлениями

* Вход: 
* * path_to_embedding_json : str - путь до Image embedding json (файла-словаря) с векторными представлениями
* * dict_with_categories_embeddings : list - список словарей {"name": label, "embedding": img_features}
* Выход: True - успешное создание | False - в противном случае


**def update_embedding_json(path_to_embedding_json : str, dict_with_categories_embeddings: list)**

> Функция для дозаписи новых размеченных данных в словарь с векторными представлениями
* Вход: 
* * path_to_embedding_json : str - путь до Image embedding json (файла-словаря) с векторными представлениями
* * dict_with_categories_embeddings : list - список словарей {"name": label, "embedding": img_features}
* Выход: True - успешная запись | False - в противном случае

# TEST.py
**Методы:**
> Функция для ручной аннотации детектированных картинок

**def label_image_pipeline(path_to_img_in_folder: str, label: str)**
* Вход: 
* * path_to_img_in_folder : str - путь до картинки
* * label : str - текстовое название товара на картинке
* Выход: dict_local_img : dict - имеет вид {"name": label, "embedding": img_features}

* Пример использования: 

```
PATH_TO_EMBEDDING_JSON = 'путь до Image embedding json'

objects = [] 
objects.append(label_image_pipeline(f"{PROJECT_PATH}/model/detected/train/1_9.jpg", 'хлеб')) 

# Картинка для тестирования 
VALID_IMAGE = f"{PROJECT_PATH}/model/detected/val/2_11.jpg" 
VALID_IMAGE = get_image(VALID_IMAGE)

# 1st method

if create_embedding_json(PATH_TO_EMBEDDING_JSON, objects):
    
    # Get all categories and embeddings
    categories, vectors = get_categories_and_embeddings(PATH_TO_EMBEDDING_JSON)

    # Fit knn
    knn = fit_knn(vectors)

    test_img_embed = vectorize_image(VALID_IMAGE)

    # test valid images
    print(k_nearest_label(categories, knn, 1, test_img_embed, show_score=True))

# 2nd method

print(cluster_img(VALID_IMAGE))

```

