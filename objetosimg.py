import cv2
import numpy as np


def preprocess_image(image):
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    
    _, threshold = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY_INV)
    
    return threshold


def detect_objects(image):
    
    threshold = preprocess_image(image)
    
    
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    
    for contour in contours:
        
        area = cv2.contourArea(contour)
        
        
        if area > 500:  
            
            x, y, w, h = cv2.boundingRect(contour)
            
            
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    return image


image_path = 'caminho/para/sua/imagem.jpg'
image = cv2.imread(image_path)


if image is None:
    print("Erro ao carregar a imagem.")
else:
    
    result_image = detect_objects(image)
    
    
    cv2.imshow('Objetos Detectados', result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()