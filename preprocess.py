
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import os



# Ön işleme adımları için oluşturulan fonksiyon
def preprocess(image_path):
    
        try:
    
            image = mpimg.imread(base_path + '/' + image_path,)
            


            # Normalizasyon      
                
            image = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                
            # Gauss
            
            image = cv2.GaussianBlur(image, (5, 5), 0)

            # Kenar Tespit  (Canny)            
            image = cv2.Canny((image*255).astype(np.uint8), 100, 200)
            
            cv2.imwrite('preprocess' + '/'+ image_path, image)
        except:
            print(image_path)
        


base_path = 'Data'

preprocess_base_path = 'preprocess'

try:
    os.mkdir(preprocess_base_path)
except: pass



# Bütün görsellerin ön işleme aşamalarından geçmesi için kullanılan döngü
for i in os.listdir(base_path):
    
    try:
        os.mkdir(preprocess_base_path + '/'+ i)
    except: pass
    
    for x in os.listdir(base_path+ '/'+ i):
        
        try:
            os.mkdir(preprocess_base_path + '/'+ i+ '/'+ x)
        except: pass
        
        for image in os.listdir(base_path+ '/'+ i + '/' + x):
            image_path =  i + '/' + x + '/' + image
            preprocess(image_path)



        
        
        

