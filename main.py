from detect import run
from union_objects import union
from model.utils import cluster_img,vectorize_image
from cluster_boxes import sort_multiline
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import os
import json
import gc
import matplotlib
matplotlib.use( 'tkagg' )

def test_and_plot(img_path:str):
    img = cv2.imread(img_path)
    pred = run(img)
    img_draw = img.copy()
    if len(pred)>0:
        plt.imshow(img)
        lines = sort_multiline(pred)
        for line_idx, line in enumerate(lines):
            for obj_idx, obj in enumerate(line):
                img_draw = cv2.rectangle(img_draw, (obj[0], obj[1]),(obj[2] , obj[3]),color=(0, 0, 255), thickness=2) # color =(Blue, Green, Red)
                cv2.putText(img_draw,  f'{line_idx}/{obj_idx}', (obj[0], obj[1]),
                             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    imgplot = plt.imshow(cv2.cvtColor(img_draw, cv2.COLOR_BGR2RGB))
    plt.show()

def crop_obj_from_img():
    delta=0.05
    img_list = glob.glob('tests_shelves/*.jpg')
    img_list += glob.glob('tests_shelves/*.jpeg')
    if not os.path.exists('./goods'):
        os.makedirs('./goods')
    for img_file in img_list:
        img_name = img_file.split('/')[-1]
        img_dir = './goods/'+img_name.split('.')[0]
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)
        img = cv2.imread(img_file)
        pred = run(img)
        if len(pred) > 0:
            lines = sort_multiline(pred)
            for line_idx, line in enumerate(lines):
                for obj_idx, obj in enumerate(line):
                    img_obj = img[obj[1]:obj[3], obj[0]:obj[2]]
                    file_name = img_dir+f'/{line_idx}_{obj_idx}.jpg'
                    cv2.imwrite(file_name,img_obj)

def detect(img:np.array):
    '''

    :param img:
    :return:
    '''
    result = []
    # детектируем объекты на изображении
    pred = run(img)
    '''
            pred - список с детектами, каждый детект - список из 4 координат в формате int, упорядоченных в порядке 
            xmin, ymin, xmax, ymax
            [[271, 659, 312, 763], [315, 661, 353, 759], [205, 219, 247, 347], [265, 0, 322, 146],....]
    '''
    if len(pred) > 0:
        lines = sort_multiline(pred)
        '''
        lines - список, внутри которого упорядоченны полки сверху вниз. 
        каждая полка - упорядоченный список объектов, упорядоченый слева на право.
        [
        [[0, 69, 34, 220], [42, 76, 76, 222], [85, 36, 119, 109], ...], 
        [[202, 462, 243, 618], [248, 466, 288, 615], [293, 468, 333, 608], ...], 
        .... ,
        ]
        '''
        for line_idx, line in enumerate(lines):
            Detection_line = []
            for obj_idx, obj in enumerate(line):
                detection = {"category": "", "group_box": obj, "obj_boxes": [obj]}
                Detection_line.append(detection)
                '''
                структура детектируемого объекта
                [[{'category': '', 'group_box': [216, 294, 271, 367], 'obj_boxes': [[216, 294, 271, 367]]},
                category:str - название категории, заполняется на этапе кластеризации
                group_box: BigBBox - большая рамка вокруг нескольких объектов одной категории. при первом детекте это 
                только объект
                obj_boxes: list(BBox) - список с рамками, каждая из которых вокруг отдельного объекта. пока объекты 
                не кластеризованны то оставляем там один объект
                '''
            result.append(Detection_line)
    return result

def test_detect(img_path:str):
    img = cv2.imread(img_path)
    res = detect(img)
    print(res)

def get_mergin_group(shelv:list):
    categorys_counter={}
    for object in shelv:
        if object['category'] != 'None':
            if object['category'] in categorys_counter:
                categorys_counter[object['category']].append(shelv.index(object))
            else:
                categorys_counter[object['category']]=[shelv.index(object)]
    merging_group=[]
    for key, value in categorys_counter.items():
        if len(value)>1:
            if could_merge(value):
                merging_group=value
                break
    if any(merging_group):
        return True, merging_group
    else:
        return False, merging_group

def could_merge(group_list):
    result = False
    for index in range(0,len(group_list)-1):
        if group_list[index+1]-group_list[index]==1:
            result = True
    return result


def infer(img:np.array ):
    '''
    эта функция вызывает функцию детекта, а потом полученные результаты кластеризует(выставляет имена) и объединяет в
    группы. По сути ее будет дергать бэкэнд
    :return:
    '''
    results = detect(img)
    # кластеризация
    final_result=[]
    img_embedings=[]
    for shelv in results:
        for detection in shelv:
            # {'category': '', 'group_box': [216, 294, 271, 367], 'obj_boxes': [[216, 294, 271, 367]]}
            crop_img = img[detection['group_box'][1]:detection['group_box'][3],
                       detection['group_box'][0]:detection['group_box'][2]]
            # получение категории вырезанного товара
            category = cluster_img(crop_img)
            detection['category'] = category
        have_group_for_mergin = True
        # объединение в группы
        while have_group_for_mergin:
            have_group_for_mergin, mergin_group = get_mergin_group(shelv)
            if have_group_for_mergin:
                new_shelv = union(shelv,mergin_group)
                shelv = new_shelv
        final_result.append(shelv)
        gc.collect()
    for shelv in results:
        for detection in shelv:
            #  was {'category': '', 'group_box': [216, 294, 271, 367], 'obj_boxes': [[216, 294, 271, 367]]}
            #  need {'category': '', 'group_box': [216, 294, 271, 367], 'obj_boxes': [{‘vector’: […],‘coordinates’: [216, 294, 271, 367]},
            #                                                                          {}
            #                                                                          {}]}
            cors = detection['obj_boxes']
            vectors = []
            for cor in cors:
                crop_img = img[cor[1]:cor[3],cor[0]:cor[2]]
                vectors.append(vectorize_image(crop_img))
            new_obj_boxes=[]
            for vec, cor in zip (vectors,cors):
                new_obj_boxes.append({'vector':vec,'coordinates':cor})
            detection['obj_boxes'] = new_obj_boxes
    # сериализация результатов в Json
    gc.collect()
    final_result = json.dumps(final_result)
    return final_result

def test_infer_merging():
    import time
    start_time = time.time()

    img=cv2.imread('./tests_shelves/1.jpg')
    res = infer(img)
    print(res)
    print("--- %s seconds ---" % (time.time() - start_time))

test_infer_merging()