import cv2
from model import MainModel
import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image

class VideoDetecter:
    def __init__(self):
        model = MainModel.Net()
        model.load_state_dict(torch.load('./pretrain_model/MaskDetection_model.pk'))
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        face_cascade.load('./haarcascades/haarcascade_frontalface_default.xml')
        # face_cascade = cv2.CascadeClassifier('haarcascade_righteye_2splits.xml')
        # face_cascade.load('./haarcascades/haarcascade_righteye_2splits.xml')
        camera = cv2.VideoCapture(0)
        while (True):
            ret, frame = camera.read()
            pic = frame.copy()
            if ret == True:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)

                for (x, y, w, h) in faces:
                    # print(frame.shape)
                    lx = x
                    rx = x+w
                    ly = y
                    ry = y+h
                    cropped = pic[max(0,ly):min(frame.shape[0], ry), max(0,lx):min(frame.shape[1],rx)]
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
                        self.draw_text(frame,'with mask', x, y - 5)
                    else:
                        color = (255, 255, 255)
                        self.draw_text(frame, 'without mask', x, y - 5)
                    img = cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

                cv2.imshow('camera', frame)
            else:
                print("no ret")

            if cv2.waitKey(int(1000 / 12)) & 0xFF == ord('q'):
                break
        camera.release()
        cv2.destroyAllWindows()
    def draw_text(self, img, text, x, y):
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)



if __name__ == "__main__":
    video_detecter = VideoDetecter()