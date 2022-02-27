import numpy as np
from sklearn.cluster import DBSCAN
from collections import defaultdict

def sort_multiline(detections,
                   max_dist: float = 0.7,
                   min_obj_per_line: int = 2) :
    '''
    кластеризует объекты на полках и упорядочевает их слева на право
    :param detections: список детектов
    :param max_dist: максимальн жакардово расстояние
    :param min_words_per_line: мин число слов в строке
    :return: список строк с детектами
    '''

    lines = _cluster_lines(detections, max_dist, min_obj_per_line)
    res = []
    for line in lines:
        res += sort_line(line)
    return res

def sort_line(detections):
    """упорядочевание строк"""
    return [sorted(detections, key=lambda det: det[2] / 2)]

def _cluster_lines(detections,
                   max_dist: float = 0.7,
                   min_words_per_line: int = 2):
    '''
    клестеризует слова в строки, вернее определяет какие слова стоят на одной строке а какие нет
    :param detections:
    :param max_dist:
    :param min_words_per_line: минимальное число слов в строке
    :return:
    '''
    num_bboxes = len(detections)
    dist_mat = np.ones((num_bboxes, num_bboxes))
    for i in range(num_bboxes):
        for j in range(num_bboxes):
            a = detections[i]
            b = detections[j]
            if a[1]> b[3] or b[1] > a[3]:
                continue
            intersection = min(a[3], b[3]) - max(a[1], b[1])
            union = (a[3] - a[1]) + (b[3] - b[1]) - intersection
            iou = np.clip(intersection / union if union > 0 else 0, 0, 1)
            dist_mat[i, j] = 1 - iou  # Jaccard distance is defined as 1-iou

    dbscan = DBSCAN(eps=max_dist, min_samples=min_words_per_line, metric='precomputed').fit(dist_mat)

    clustered = defaultdict(list)
    for i, cluster_id in enumerate(dbscan.labels_):
        if cluster_id == -1:
            continue
        clustered[cluster_id].append(detections[i])

    res = sorted(clustered.values(), key=lambda line: [det[3] / 2 for det in line])
    return res