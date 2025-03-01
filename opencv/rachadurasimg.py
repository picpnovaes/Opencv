import cv2
import numpy as np


def preprocess_image(image):
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    
    edges = cv2.Canny(blurred, 50, 150)
    
    return edges


def detect_cracks(image):
    
    edges = preprocess_image(image)
    
    
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    
    for contour in contours:
        
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        
        cv2.drawContours(image, [approx], -1, (0, 255, 0), 2)
    
    return image


image_path = r'Exemplo\Pasta\Imagem.jpg'
image = cv2.imread(image_path)


if image is None:
    print("Erro ao carregar a imagem.")
else:
    
    result_image = detect_cracks(image)
    
    
    cv2.imshow('Rachaduras Detectadas', result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()