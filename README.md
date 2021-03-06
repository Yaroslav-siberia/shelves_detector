# Analysis of goods on store shelves
Для решения задачи детектирования объектов на полках было решено использовать нейронную сеть семейства YOLOV5. 
В качестве данных для обучения был выбран SKU-110K Dataset от Amazon, содержащий более 11 тысяч изображений и более 1.7 млн. размеченных объектов.  
После тестирования была выбрана нейронная сеть yolov5m (medium) как наиболее оптимальная. Данная сеть показала следующие результаты:  
Precission - 0.93 -  доля объектов, названных классификатором положительными и при этом действительно являющимися положительными.  
Recall - 0.91 - доля объектов положительного класса из всех объектов положительного класса нашел алгоритм.  
mAP - 0.91 - среднее по всем детектируемым объектам значение показтеля IOU. IOU (Intersection over Union) - отношение рамки детектирования к эталонной рамке объекта.  

Сети семейства YOLOV5, но имеющие более сложную архитектуру (yolov5l, yolov5x), показали результат незначительно лучше, но в виду большего объема параметров ощутимо меньшую скорость работы.

<p align="left">
  <img src="https://github.com/Yaroslav-siberia/shelves_detector/blob/main/tests_shelves/1.jpg">
  <img src="https://github.com/Yaroslav-siberia/shelves_detector/blob/main/tests_results/1.jpg">
</p>
  
В директории shelves_detector/research/ можно найти результаты проверок разных гипотез по кластеризации задетектированных товаров

# TODO  
Внедрить Metric Learning для кластеризации объектов

# Detecting objects

> /exp/weights/best.pt содержит сериализованную нейронную сеть детектирования

## main.py

**Mетоды:**

**def infer(img:np.array)**

> Функция принимает на вход картинку формате np.array. Используется при первом сканировании фотографии. Когда мы в первый раз фотку загрузили, сетка соответственно задетектила и что смогла- распознала. Возвращается структура с выделениями  
> Вызывать после загрузки фотки 

-   Вход: img : np.array - картинка
-   Выход: json : список, содержащий полки с товарами. каждый товар представлен в виде словаря dict
-   Пример структуры, описывающей 1 единицу товара:
    > {"category": "cat1", "group_box": [xmin,ymin,xmax,ymax], "obj_boxes": [[xmin,ymin,xmax,ymax]]}  
    > "category" - категория товара которую смогли определить,  
    > "group_box" - рамка вокруг группы товаров  
    > "obj_boxes" - список рамок вокруг каждой единицы товара
-   Пример структуры, описывающей фотографию:
    > [  
    > [  
    > {"category": "cat1", "group_box": [27, 36, 504, 243], "obj_boxes": [[[27, 83, 100, 233]], [[105, 83, 187, 235]], [[197, 86, 273, 237]], [[278, 74, 357, 240]], [[362, 88, 438, 241]], [[443, 36, 504, 243]]]},  
    > {"category": "cat2": [558, 42, 787, 253], "obj_boxes": [[[558, 42, 623, 248]], [[625, 42, 682, 252]], [[685, 48, 737, 253]], [[737, 56, 787, 252]]]},  
    > {"category": "cat3", "group_box": [790, 48, 1202, 276], "obj_boxes": [[[790, 81, 847, 255]], [[852, 119, 914, 261]], [[917, 123, 986, 264]], [[990, 77, 1056, 272]], [[1061, 66, 1127, 272]], [[1129, 48, 1202, 276]]]}  
    > ]  
    > ]

# Grouping objects

## union_objects.py

**Методы**

**def union(shelve:list,mergen_list:list):**

> Функция пересобирает конструкцию полки, объединяя товары в группу.
> Принимает полку и список индексов(порядковый номер на полке) которые надо объединить в одну группу.
> Применяется когда сетка задетектила все по отдельности и мы хотим объединить в группу. соответственно пользователь должен выделить несколько объектов и нажать кнопку объединить.соответвенно надо отправить полку (структуру) на которой находятся объединяемые объекты и индексы объектов что надо объединить.

-   Вход: shelve : list - полка - список объектов на полке.
-   Вход: mergen_list : list - список индексов товаров которые надо объединить.
-   Выход: shelve : list - новая конфигурация полки.

-   Пример входных данных, для объединения товаров в группу:
    > пример полки  
    > [  
    > {"category": "cat1", "group_box": [27, 36, 504, 243], "obj_boxes": [[[27, 83, 100, 233]], [[105, 83, 187, 235]], [[197, 86, 273, 237]], [[278, 74, 357, 240]], [[362, 88, 438, 241]], [[443, 36, 504, 243]]]},  
    > {"category": "cat2": [558, 42, 787, 253], "obj_boxes": [[[558, 42, 623, 248]], [[625, 42, 682, 252]], [[685, 48, 737, 253]], [[737, 56, 787, 252]]]},  
    > {"category": "cat3", "group_box": [790, 48, 1202, 276], "obj_boxes": [[[790, 81, 847, 255]], [[852, 119, 914, 261]], [[917, 123, 986, 264]], [[990, 77, 1056, 272]], [[1061, 66, 1127, 272]], [[1129, 48, 1202, 276]]]}  
    > ]  
    > Пример списка индексов  
    > [0,1,2]

## Clustering and vectorizing objects

## model/utils.py

**def cluster_img(img : np.array)**

> Функция принимает на вход картинку и по порогу KNN_THRESHOLD относит к определенному кластеру  
> P.S. на вход подается уже вырезанный товар по координатам из obj_boxes.  
> Вызывается в первой функции infer автоматически чтобвы для картинки получить метку класса(категории) если необходимо - вызывать самим в ручную
> Для вырезания товара использовать croped_image = image[ymin:ymax, xmin:xmax]

-   Вход: img : np.array - картинка
-   Выход: label: str | 'None'

**def vectorize_image(img : np.array)**

> Функция принимает на вход картинку и возвращает векторное предствление. Используется для получения векторного представления товара.
> Вызывается чтобы получить векторное представление категории. ее результат потом уходит в update_image-embeding
> P.S. на вход подается уже вырезанный товар по координатам из obj_boxes.  
> Для вырезания товара использовать croped_image = image[ymin:ymax, xmin:xmax]

-   Вход: img : np.array - картинка
-   Выход: img_embedding: np.array

**def update_image-embeding.(path_to_embedding_json : str, dict_with_categories_embeddings: list)**

> Функция для дозаписи новых размеченных данных в словарь с векторными представлениями. 
> Используется чтобы добавить новый товар(категорию) и его векторное представление в базу.

* Вход: 
* * path_to_embedding_json : str - путь до Image embedding json (файла-словаря) с векторными представлениями
* * dict_with_categories_embeddings : list - список словарей {"name": label, "embedding": img_features}
* Выход: True - успешная запись | False - в противном случае


**def update_knn(PATH_TO_EMBEDDING_JSON: str)**

> Функция перечитывает нашу базу и сразу переобучает knn - механизм кластеризации. Тем самым система учит новый товар. 
> Вызывать после того как добавили некоторое количество новых товаров в базу.  
> ВНИМАНИЕ!! процедура не быстра потому после каждой новой единицы товара запускать не стоит
* Вход: 
* * path_to_embedding_json : str - путь до Image embedding json (файла-словаря) с векторными представлениями
* Выход: True - успешное переобучение | False - в противном случае
