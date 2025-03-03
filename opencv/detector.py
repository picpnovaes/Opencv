import cv2
import numpy as np


def preprocess_image_objects(image):
    """Pré-processamento para detecção de objetos."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, threshold = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY_INV)
    return threshold


def detect_objects(image):
    """Detecta objetos na imagem e desenha retângulos ao redor deles."""
    threshold = preprocess_image_objects(image)
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 500:  
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    return image


def preprocess_image_cracks(image):
    """Pré-processamento para detecção de rachaduras."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    return edges


def detect_cracks(image):
    """Detecta rachaduras na imagem e desenha contornos ao redor delas."""
    edges = preprocess_image_cracks(image)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        cv2.drawContours(image, [approx], -1, (0, 0, 255), 2)  # Vermelho para rachaduras
    
    return image


# Caminho da imagem
image_path = 'caminho/para/sua/imagem.jpg'
image = cv2.imread(image_path)

if image is None:
    print("Erro ao carregar a imagem.")
else:
    # Aplicar ambas as detecções na mesma imagem
    image_with_objects = detect_objects(image.copy())
    final_image = detect_cracks(image_with_objects)

    # Exibir resultado
    cv2.imshow('Detecção de Objetos e Rachaduras', final_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
