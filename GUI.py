

from PIL import Image, ImageTk
import tkinter as tk
import tkinter.filedialog
from Detection.VideoDetection import VideoDetecter
from Detection.Img_detection import Img_detector

# 创建窗口 设定大小并命名
window = tk.Tk()
window.title('Mask detection GUI')
window.geometry('1200x1000')
global img_png
var = tk.StringVar()
filename = tk.StringVar()

def predict_image():
    global img_png
    var.set('Predicted')
    img_detector = Img_detector(filename.get())
    Img = Image.fromarray(img_detector.getPredictImg())
    img_png = ImageTk.PhotoImage(Img)
    label_Img = tk.Label(window, image=img_png)
    label_Img.pack()


def choose_file():
    selectFileName = tkinter.filedialog.askopenfilename(title='Choose your file')
    filename.set(selectFileName)
    # print(var.get())

def openCamera():
    face_detector = VideoDetecter()


Label_Show = tk.Label(window,
    textvariable=var,
    bg='yellow', font=('Arial', 12), width=15, height=2)
Label_Show.pack()
# choose your file
btn_Choose = tk.Button(window,
    text='choose your file',
    width=20, height=2,
    command=choose_file)
btn_Choose.pack()

# predict your image
btn_Show = tk.Button(window,
    text='Predict',
    width=15, height=2,
    command=predict_image)
btn_Show.pack()

# open camera
btn_Show = tk.Button(window,
    text='Open Cemera',
    width=15, height=2,
    command=openCamera)
btn_Show.pack()

# run windows
window.mainloop()
