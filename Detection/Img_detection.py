import cv2
from model import MainModel
import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image

class Img_detector:
    def __init__(self, filename):
        model = MainModel.Net()
        model.load_state_dict(torch.load('./pretrain_model/MaskDetection_model.pk'))
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        face_cascade.load('./haarcascades/haarcascade_frontalface_default.xml')
        img = cv2.imread(filename)
        # img = Image.fromarray(img)
        pic = img.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            lx = x
            rx = x + w
            ly = y
            ry = y + h
            cropped = pic[max(0, ly):min(img.shape[0], ry), max(0, lx):min(img.shape[1], rx)]
            # cv2.imwrite("1.jpg",cropped)
            cropped = Image.fromarray(cropped)

            resize = transforms.Compose([
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            cropped = resize(cropped)
            output = model(cropped.unsqueeze(0))
            _, predicted = torch.max(output.data, 1)
            if predicted.item() == 1:
                color = (255, 0, 0)
                self.draw_text(img, 'with mask', x, y - 5)
            else:
                color = (255, 255, 255)
                self.draw_text(img, 'without mask', x, y - 5)
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        self.img = img

    def draw_text(self, img, text, x, y):
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

    def getPredictImg(self):
        return cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)

