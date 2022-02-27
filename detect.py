import os
import glob
import cv2
import torch
from PIL import Image
from yolov5 import YOLOv5
import torchvision.transforms as T

import numpy as np

# Model path
shelves_model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "exp/weights/best.pt")


# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = YOLOv5(shelves_model_path, device)

# allocating color
color = (0, 0, 255)  # red

torch.manual_seed(17)

trans = T.Compose([
    T.Resize(640),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@torch.no_grad()
def run(img:np.array):
    result = []
    img = img.astype(np.uint8)  # dont touch this
    img = Image.fromarray(img)
    tens = trans(img)
    tens = tens.unsqueeze(0)
    tens = tens.to(device)
   # size = max(list(img.shape))
    predictions = model.predict(img)#, size=size)
    predictions = predictions.pred[0]
    if len(predictions) > 0:  # если обнаружили что-то
        boxes = predictions[:, :4]
        scores = predictions[:, 4].tolist()
        categories = predictions[:, 5].tolist()
        for box,score in zip(boxes,scores):
            if score>0.6:
                box = [int(a) for a in box]
                result.append(box)
    return result



def allocating_boxes(img, boxes_list, color = (0, 255, 0), delta=0):
    """
    функция выделяет в кадре прямоугольники по указанным координатам с указанным цветом
    :param frame: - кадр зоны
    :param boxes_list:  - список боксов которые надо выделить
    :param color: - цвет которым надо выделять
    :param delta: - сдвиг рамки
    :return: new_frame - кадр зоны с выделенными нарушениями или соблюденирями правил
    """
    delta *= 3
    new_img = img.copy()
    for box in boxes_list:
        new_img = cv2.rectangle(new_img, (box[0] - delta, box[1] - delta),
                                  (box[2] + delta, box[3] + delta),
                                  color=color, thickness=2)
    return new_img


def test():
    img_list = glob.glob('./tests_shelves/*.jpg')
    img_list += glob.glob('./tests_shelves/*.jpeg')
    if not os.path.exists('./tests_results'):
        os.makedirs('./tests_results')
    for img_file in img_list:
        img = cv2.imread(img_file, cv2.COLOR_BGR2RGB)
        pred = run(img)
        print(pred)
        if len(pred)>0:
            img_res = allocating_boxes(img,pred)
            img_name = img_file.split('/')[-1]
            cv2.imwrite(f'./tests_results/{img_name}',img_res)



#test()

'''
возвращаемое значение - список коодинатов задетектированных объектов

[[271, 659, 312, 763], 
[315, 661, 353, 759], 
[205, 219, 247, 347], 
[265, 0, 322, 146] 

'''
