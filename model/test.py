from unicodedata import category
from utils import vectorize_image, get_image, NumpyArrayEncoder, update_embedding_json, k_nearest_label, fit_knn, get_categories_and_embeddings, create_embedding_json, cluster_img
import json
import os

PROJECT_PATH = os.path.join(os.path.dirname(
    os.path.realpath(__file__)), "detected/")

print(f'Project path: {PROJECT_PATH}')
PATH_TO_EMBEDDING_JSON = os.path.join(os.path.dirname(
    os.path.realpath(__file__)), "img_embeddings.json")


# List to serialize annotated images
all = []


def label_image_pipeline(path_to_img_in_folder: str, label: str):
    '''
        Функция принимает путь до картинки

        Преобразует в np.array

                Получает векторизированное представление

                Сохраняет структуру в словарь


        '''
    img = get_image(path_to_img_in_folder)
    img_features = vectorize_image(img)
    dict_local_img = {"name": label, "embedding": img_features}
    return dict_local_img


# Train
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/2_0_0.jpg", 'белуга ликер'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/2_0_1.jpg", 'белуга ликер'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/2_0_4.jpg", 'ледофф белый'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/2_0_5.jpg", 'деревенька коньяк'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/2_0_6.jpg", 'деревенька водка'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/2_0_7.jpg", 'деревенька водка'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/2_0_8.jpg", 'стужа'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/2_0_9.jpg", 'стужа'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/2_0_10.jpg", 'стужа'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/2_0_11.jpg", 'стужа'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/2_0_12.jpg", 'газпром'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/2_0_13.jpg", 'газпром'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/2_0_14.jpg", 'урожай'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/2_0_15.jpg", 'урожай'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/2_0_16.jpg", 'олень'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/2_0_17.jpg", 'олень'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/2_0_18.jpg", 'олень'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/2_0_21.jpg", 'березка'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/2_0_22.jpg", 'березка'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/2_0_23.jpg", 'березка'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/2_0_24.jpg", 'хаски'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/2_0_25.jpg", 'хаски'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/2_0_27.jpg", 'пять озер'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/2_1_2.jpg", 'тундра бело синяя'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/2_1_4.jpg", 'тундра бело зеленая'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/2_1_6.jpg", 'тундра бело желтая'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/2_1_11.jpg", 'кизляр'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/2_1_12.jpg", 'кизляр'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/2_1_13.jpg", 'медная лошадь'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/2_1_14.jpg", 'медная лошадь'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/2_1_15.jpg", 'медная лошадь'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/2_1_16.jpg", 'белая сова'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/2_1_17.jpg", 'старая казань'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/2_1_18.jpg", 'царь кедр темная'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/2_1_19.jpg", 'царь кедр светлая'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/2_1_20.jpg", 'хлебная'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/2_1_21.jpg", 'русская валюта'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/2_1_22.jpg", 'белая сова'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/2_1_23.jpg", 'медная лошадь'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/2_1_25.jpg", 'хортиця'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/2_1_26.jpg", 'хортиця'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/2_1_27.jpg", 'хортиця'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/2_1_28.jpg", 'хортиця'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/2_1_29.jpg", 'хортиця'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/2_2_2.jpg", 'чача'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/2_2_3.jpg", 'чача'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/2_2_4.jpg", 'чача'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/2_2_5.jpg", 'ледофф синий'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/2_2_6.jpg", 'ледофф синий'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/2_2_7.jpg", 'ледофф белый'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/2_2_8.jpg", 'ледофф белый'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/2_2_9.jpg", 'ледофф белый'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/2_2_10.jpg", 'ледофф красный'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/2_2_11.jpg", 'ледофф красный'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/2_2_12.jpg", 'ледофф красный'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/2_2_13.jpg", 'ледофф красный'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/2_2_14.jpg", 'беленькая'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/2_2_15.jpg", 'беленькая'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/2_2_16.jpg", 'беленькая'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/2_2_17.jpg", 'беленькая'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/2_2_18.jpg", 'беленькая'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/2_2_19.jpg", 'беленькая'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/2_2_20.jpg", 'беленькая'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/2_3_0.jpg", 'чернослив'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/2_3_1.jpg", 'чернослив'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/2_3_2.jpg", 'клюква'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/2_3_3.jpg", 'клюква'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/2_3_4.jpg", 'клюква'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/2_3_5.jpg", 'клюква'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/2_3_6.jpg", 'клюква'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/2_3_7.jpg", 'клюква'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/2_3_8.jpg", 'клюква'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/2_3_9.jpg", 'деревенька коньяк'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/2_3_11.jpg", 'хлебная'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/2_3_12.jpg", 'деревенька водка'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/2_3_13.jpg", 'царь кедр светлая'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/2_3_14.jpg", 'деревенька водка'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/2_3_15.jpg", 'царь кедр темная'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/2_3_17.jpg", 'серебряная гора'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/2_3_19.jpg", 'урожай'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/2_3_20.jpg", 'серебряная гора'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/2_3_21.jpg", 'урожай'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/2_3_23.jpg", 'урожай'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/2_3_26.jpg", 'олень'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/2_3_28.jpg", 'олень'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/2_3_29.jpg", 'русский север'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/2_3_31.jpg", 'русский север'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/2_3_33.jpg", 'русский север'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/2_3_35.jpg", 'русский север'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/2_3_34.jpg", 'царь красный'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/2_3_36.jpg", 'царь красный'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/2_3_37.jpg", 'царь белый'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/2_3_38.jpg", 'царь белый'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/2_4_0.jpg", 'манчестер джин'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/2_4_2.jpg", 'хуч'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/2_4_3.jpg", 'хуч'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/2_4_4.jpg", 'хуч'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/2_4_5.jpg", 'жигулевское 1978'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/2_4_6.jpg", 'жигулевское 1978'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/2_4_7.jpg", 'хайникен'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/2_4_8.jpg", 'хайникен'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/2_4_10.jpg", 'амстел'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/2_4_11.jpg", 'амстел'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/2_4_12.jpg", 'крушовице светлое'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/2_4_13.jpg", 'крушовице светлое'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/2_4_14.jpg", 'гессер'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/2_4_15.jpg", 'гессер'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/2_4_16.jpg", 'охота крепкое'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/2_4_18.jpg", 'балтика 7'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/2_4_19.jpg", 'балтика 7'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/6_1_0.jpg", 'талка'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/6_1_1.jpg", 'талка'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/6_1_2.jpg", 'талка'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/6_1_3.jpg", 'зеленая марка'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/6_1_4.jpg", 'зеленая марка'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/6_1_5.jpg", 'зеленая марка'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/6_1_6.jpg", 'зеленая марка'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/6_1_7.jpg", 'зеленая марка'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/6_1_8.jpg", 'пять озер'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/6_1_9.jpg", 'пять озер'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/6_1_10.jpg", 'пять озер'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/6_1_11.jpg", 'пять озер'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/6_1_12.jpg", 'путинка'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/6_1_13.jpg", 'путинка'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/6_1_14.jpg", 'урожай'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/6_1_15.jpg", 'урожай'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/6_1_17.jpg", 'хлебная'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/6_2_2.jpg", 'царская'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/6_2_3.jpg", 'тундра бело синяя'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/6_2_4.jpg", 'тундра бело зеленая'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/6_2_5.jpg", 'тундра бело зеленая'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/6_2_7.jpg", 'ледофф синий'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/6_2_8.jpg", 'ледофф синий'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/6_2_9.jpg", 'ледофф лемон'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/6_2_10.jpg", 'ледофф бело красный'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/6_2_11.jpg", 'ледофф красный'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/6_2_12.jpg", 'ледофф красный'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/6_2_13.jpg", 'тельняжка'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/6_2_14.jpg", 'русская валюта'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/6_2_15.jpg", 'русская валюта'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/6_2_16.jpg", 'фински'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/6_2_17.jpg", 'фински'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/6_3_0.jpg", 'пчелка'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/6_3_1.jpg", 'пчелка'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/6_3_13.jpg", 'топаз'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/6_3_14.jpg", 'топаз'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/6_3_21.jpg", 'тельняжка'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/6_5_0.jpg", 'балтика 7'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/6_5_1.jpg", 'балтика 7'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/6_5_6.jpg", 'манчестер джин'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/9_0_13.jpg", 'скотч терьер'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/9_0_14.jpg", 'скотч терьер'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/9_1_0.jpg", 'царская'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/9_1_1.jpg", 'царская'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/9_1_4.jpg", 'тундра бело синяя'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/9_1_7.jpg", 'тундра бело синяя'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/9_1_5.jpg", 'тундра белая'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/9_1_8.jpg", 'тундра белая'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/9_1_6.jpg", 'тундра бело желтая'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/9_1_10.jpg", 'ледофф синий'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/9_1_11.jpg", 'ледофф синий'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/9_1_12.jpg", 'ледофф синий'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/9_1_14.jpg", 'ледофф лемон'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/9_1_15.jpg", 'ледофф бело красный'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/9_1_16.jpg", 'ледофф синий'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/9_1_17.jpg", 'царь кедр светлая'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/9_1_18.jpg", 'царь кедр темная'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/9_1_19.jpg", 'хлебная'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/9_1_23.jpg", 'старый кенигсберг'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/9_2_0.jpg", 'хаски'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/9_2_1.jpg", 'хаски'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/9_2_2.jpg", 'сибитер'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/9_2_3.jpg", 'сибитер'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/9_2_4.jpg", 'пять озер'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/9_2_5.jpg", 'пять озер'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/9_2_6.jpg", 'пять озер'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/9_2_7.jpg", 'пять озер'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/9_2_8.jpg", 'пять озер'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/9_2_9.jpg", 'пять озер'))
all.append(label_image_pipeline(
    f"{PROJECT_PATH}/train/9_2_10.jpg", 'пять озер'))


# Картинка для тестирования


VALID_IMAGE = f"{PROJECT_PATH}/val/2_0_26.jpg"
VALID_IMAGE = get_image(VALID_IMAGE)


# 1st method
# Main
if create_embedding_json(PATH_TO_EMBEDDING_JSON, all):
    # Get all categories and embeddings
    categories, vectors = get_categories_and_embeddings(PATH_TO_EMBEDDING_JSON)

    # Fit knn
    knn = fit_knn(vectors)

    test_img_embed = vectorize_image(VALID_IMAGE)

    # test valid images
    print(k_nearest_label(categories, knn, 1, test_img_embed, show_score=True))

# 2nd method

print(cluster_img(VALID_IMAGE))
