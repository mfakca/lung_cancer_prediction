import tkinter as tk
from tkinter import filedialog
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
from tensorflow.keras import Model 
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.preprocessing import image
import os 

class Application(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Örnek Uygulama")
        
        self.create_upload_section()
        self.create_preprocess_section()
        self.create_model_section()
        self.create_result_section()
        self.classes_dir = ["Kanser (Adenocarcinoma)", "Kanser (Large cell carcinoma)", "Kanser Değil", "Kanser (Squamous cell carcinoma)"]
        self.model_path = ['ct_vgg_best_model.hdf5','ct_incep_best_model.hdf5','ct_effnet_best_model.hdf5']
        
        

    def create_upload_section(self):
        frame =  tk.LabelFrame(self, text="1. Görsel Yükle:", pady=10)
        tk.Label(frame, text="Görsel Yükle:").pack(side=tk.LEFT)
        tk.Button(frame, text="Upload", command=self.upload_action).pack(side=tk.LEFT)
        frame.pack(fill=tk.X)

    def create_preprocess_section(self):
        frame = tk.LabelFrame(self, text="2. Ön İşleme Seçenekleri:", pady=10)
        self.preprocess_vars = [tk.BooleanVar() for _ in range(3)]
        tk.Checkbutton(frame, text=f"Normalizasyon", variable=self.preprocess_vars[0]).grid(row=1, column=1, sticky=tk.W)
        tk.Checkbutton(frame, text=f"Gauss Filtresi", variable=self.preprocess_vars[1]).grid(row=1, column=2, sticky=tk.W)
        tk.Checkbutton(frame, text=f"Kenar Tespit", variable=self.preprocess_vars[2]).grid(row=1, column=3, sticky=tk.W)
        
        

        frame.pack(fill=tk.X)

    def create_model_section(self):
        frame = tk.LabelFrame(self, text="3. Model Seçimi:", pady=10)
        self.model_vars = tk.IntVar()
        
            
        tk.Radiobutton(frame, text=f"VGG16", variable=self.model_vars, value = 0).grid(row=1, column=1, sticky=tk.W)
        tk.Radiobutton(frame, text=f"Inception", variable=self.model_vars, value = 1).grid(row=1, column=2, sticky=tk.W)
        tk.Radiobutton(frame, text=f"Efficient", variable=self.model_vars, value = 2).grid(row=1, column=3, sticky=tk.W)
        
        frame.pack(fill=tk.X)

    def create_result_section(self):
        
        self.predicted_result = tk.StringVar()
        frame =  tk.LabelFrame(self, text="4. Sonuç:", pady=10)
        
        
        
        tk.Button(frame, text="Görseli Göster!", command=  self.show_image).grid(row=0, column=0, padx= 7, pady=5)
        tk.Button(frame, text="Değişimi Göster!", command =  self.preprocess_image).grid(row=0, column=1, padx= 7, pady=5)
        tk.Button(frame, text="Tahmin Et!", command=self.predict_action).grid(row=0, column=2, padx= 7, pady=5)
        frame.pack(fill=tk.X)
        
    def show_image(self):
        
        self.image = mpimg.imread(self.filename)
        plt.imshow(self.image, cmap='gray')
        plt.axis('off')
        plt.show()
        
        
        
        
    
        
    def preprocess_image(self):

        self.image = mpimg.imread(self.filename)


        # Normalizasyon
        if self.preprocess_vars[0].get() == True:
            
            self.image = cv2.normalize(self.image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            
        # Gauss
        if self.preprocess_vars[1].get() == True:
        
            self.image = cv2.GaussianBlur(self.image, (5, 5), 0)

        # Kenar Tespit  (Canny)
        if self.preprocess_vars[2].get() == True:    
            
            self.image = cv2.Canny((self.image*255).astype(np.uint8), 100, 200)
          
            
        plt.imshow(self.image, cmap='gray')
        plt.axis('off')
        plt.show()
        


    
    

    def upload_action(self):
        self.filename = filedialog.askopenfilename()
        print(f'Selected file: {self.filename}')  # Yükleme işlemini burada gerçekleştirebilirsiniz.

    def predict_action(self):
        
        
        try:
            os.mkdir('predict')
        except: pass
        
        
        try:       
        
            plt.imsave('predict/predict.png',self.image)
        
        except:
            self.image = mpimg.imread(self.filename)
            plt.imsave('predict/predict.png',self.image)
            
            

        #self.model_path[self.model_vars.get()]
        
        self.model = load_model(self.model_path[self.model_vars.get()])
        
        self.image = image.load_img('predict/predict.png', target_size=(350, 350))  

        # Normalizing Image
        norm_img = image.img_to_array(self.image) / 255

        # Converting Image to Numpy Array
        input_arr_img = np.array([norm_img])

        # Getting Predictions
        pred = np.argmax(self.model.predict(input_arr_img))
        
        tk.messagebox.showinfo("Sonucunuz",  'Sonuç: ' + self.classes_dir[pred]) 
        # Printing Model Prediction
        #self.predicted_result.set =  self.classes_dir[pred]

if __name__ == "__main__":
    app = Application()
    app.resizable(width=False, height=False)
    app.mainloop()
