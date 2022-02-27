import math
import gc

def count_new_bbox(mergin_bboxes:list):
    '''
    высчитывает рамку, которая будет выделять группу однотипных товаров. соответственно работает только group_box.
    чтобы можно было объединять группы товаров друг с другом, одиночные и группу с одиночным
    :param mergin_bboxes: список bbox с координатами
    :return: new box
    '''
    new_box=[math.inf,math.inf, 0, 0]  #xmin,ymin,xmax,ymax
    for box in mergin_bboxes:
        if box[0] < new_box[0]:  # ищем минимальный xmin
            new_box[0] = box[0]
        if box[1] < new_box[1]:  # ищем минимальный ymin
            new_box[1] = box[1]
        if box[2] > new_box[2]:  # ищем максимальный  xmax
            new_box[2] = box[2]
        if box[3] > new_box[3]:  # ищем максимальный  ymax
            new_box[3] = box[3]
    return new_box

def merge_objects_on_shelv(obj_group:list):
    group_box_list=[x['group_box'] for x in obj_group]
    new_group_box = count_new_bbox(group_box_list)
    new_obj_boxes = [x['obj_boxes'] for x in obj_group]

    merged_object = {'category': obj_group[0]['category'], 'group_box': new_group_box, 'obj_boxes': new_obj_boxes}
    return merged_object

def get_group_index(number,groups):
    index = 0
    for i in range(0,len(groups)):
        if number in groups[i]:
            index = i
            break
    return index

def index_in_meging_group(number,groups):
    result = False
    for group in groups:
        if number in group:
            result = True
            break
    return result

def union(shelve:list,mergen_list:list):
    '''
    функция получает на вход струтуру, описывающую объекты на полке, и структуру, описывающую что надо объединить
    :param shelve: описание товаров на полке
    [{'category': 'cat1', 'group_box': [216, 294, 271, 367], 'obj_boxes': [[216, 294, 271, 367]]},
    {'category': 'cat2', 'group_box': [290, 294, 300, 367], 'obj_boxes': [[290, 294, 300, 367]]},...]
    :param mergen_list: описание какие товары надо объединить. указываются индексы на полке. соответственно объединить
    можно только на одной полке (потому и только полка подается на вход) и только рядом стоящие, например [1,2,3].
    а вариант типа : [1,3,6] объединить нельзя потому что между ними другие товары.
    так же работает на всякий случай когда у нас 1 товар стоит с разрывом например [1,2,3,6,7,8] -
    тогда объединит в 2 группы [1,2,3] и [6,7,8]
    [1,2,3] - надо в одну группу объединить товары с индексами 1,2,3
    :return: new_shelve возвращает новую структуру полки
    '''
    mergen_list.sort()# отсортировали на всякий случай
    # разбивыаем на групки на подобии [1,2,3,6,7,8] => [[1,2,3], [6,7,8]]
    groups=[]
    index_start=0
    index_end=0
    for i in range(0,len(mergen_list)-1):
        if mergen_list[i+1]-mergen_list[i]==1:
            index_end=i+1
        else:
            group = mergen_list[index_start:index_end+1]
            if any(group):
                groups.append(group)
            index_start=index_end+1
    if index_start != index_end:
        group = mergen_list[index_start:index_end+1]
        if any(group):
            groups.append(group)
    # вот тут мы сделали преобразование [1,2,3,6,7,8] => [[1,2,3], [6,7,8]] и лежит groups
    # теперь собрали сами характеристики объектов которые надо обэединять, аналогично по группам
    obj_groups_for_merging=[]
    for group in groups:
        obj_group=[]
        for i in group:
            obj_group.append(shelve[i])
        if any(obj_group):
            obj_groups_for_merging.append(obj_group)
    # в obj_groups_for_merging лежат списки, внутри списка уже не индексы а сами объекты. каждый список обединится в один объект
    # теперь для каждой группы получаем объединенный объект
    merged_objects = []
    for obj_group in obj_groups_for_merging:
        merged_object = merge_objects_on_shelv(obj_group)
        merged_objects.append(merged_object)
    new_shelve=[]
    # добавили объекты с полок которые стояли до первой группы объединения
    index = 0
    while index < len(shelve)-1:
        #if index in mergen_list:
        if index_in_meging_group(index,groups):
            #get group index
            obj_index = get_group_index(index, groups)
            # get selected object
            merged_object = merged_objects[obj_index]
            new_shelve.append(merged_object)
            index = groups[obj_index][-1]
        else:
            new_shelve.append(shelve[index])
        index +=1
    gc.collect()
    new_shelve=list(tuple(new_shelve))
    return new_shelve