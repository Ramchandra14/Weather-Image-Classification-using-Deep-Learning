# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 12:31:39 2022

@author: ramch
"""

import tkinter as tk
from PIL import ImageTk, Image
from tkinter import filedialog
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array

#path="model_VGG16.h5"
path='model_VGG16_1.h5'

#loading the model to be used
from keras.models import load_model 
# load model
model = load_model(path)

def load_img():
    global img, image_data
    for img_display in frame.winfo_children():
        img_display.destroy()

    image_data = filedialog.askopenfilename(initialdir="/", title="Choose an image",
                                       filetypes=(("all files", "*.*"), ("png files", "*.png")))
    basewidth = 300 # Processing image for displaying
    img = Image.open(image_data)
    wpercent = (basewidth / float(img.size[0]))
    hsize = int((float(img.size[1]) * float(wpercent)))
    img = img.resize((basewidth, hsize), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)
    file_name = image_data.split('/')
    panel = tk.Label(frame, text= str(file_name[len(file_name)-1]).upper()).pack()
    panel_image = tk.Label(frame, image=img).pack()
    
def classify():
    original = Image.open(image_data)
    original = original.resize((224, 224), Image.ANTIALIAS)
    numpy_image = img_to_array(original)
    image_batch = np.expand_dims(numpy_image, axis=0)
    
    label = model.predict(image_batch)
    table = tk.Label(frame, text="").pack()
    
    l={"dew":label[0][0],"fogsmog":label[0][1],"frost":label[0][2],"glaze":label[0][3],"hail":label[0][4],"lightning":label[0][5],
       "rain":label[0][6],"rainbow":label[0][7],"rime":label[0][8],"sandstorm":label[0][9],"snow":label[0][10]}
    val=label.max()

    #def get_key(val):
    for key, value in l.items():
          if val == value:
              result = tk.Label(frame,text=str(key),font=("", 30)).pack()
     
        #return "key doesn't exist"
   
         
root =tk.Tk()
root.title('Weather Image Classifier')
root.resizable(False, False)
tit = tk.Label(root, text="Weather Image Recognition", padx=25, pady=6, font=("", 30)).pack()
canvas = tk.Canvas(root, height=500, width=500, bg='grey')
canvas.pack()
frame = tk.Frame(root, bg='white')
frame.place(relwidth=0.8, relheight=0.8, relx=0.1, rely=0.1)
chose_image = tk.Button(root, text='Choose Image',
                        padx=35, pady=10,
                        fg="white", bg="black", command=load_img)
chose_image.pack(side=tk.LEFT)
class_image = tk.Button(root, text='Classify Image',
                        padx=35, pady=10,
                        fg="white", bg="black", command=classify)
class_image.pack(side=tk.RIGHT)
root.mainloop()       
