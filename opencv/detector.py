import cv2
import numpy as np

def preprocess_image_objects(image):
    """Pré-processamento para detecção de objetos."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, threshold = cv2.threshold(blurred, 120, 255, cv2.THRESH_BINARY_INV)
    
    # Removendo ruídos pequenos
    kernel = np.ones((3, 3), np.uint8)
    threshold = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel, iterations=2)

    return threshold

def detect_objects(image):
    """Detecta objetos na imagem e desenha retângulos ao redor deles."""
    threshold = preprocess_image_objects(image)
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if 500 < area < 10000:  # Evita pequenos ruídos e grandes falsos positivos
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Verde para objetos
            cv2.putText(image, "Objeto", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return image

def preprocess_image_cracks(image):
    """Pré-processamento para detecção de rachaduras."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    # Melhorando a segmentação das rachaduras
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    return edges

def detect_cracks(image):
    """Detecta rachaduras na imagem e desenha contornos ao redor delas."""
    edges = preprocess_image_cracks(image)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) > 100:  # Evita detecções falsas
            cv2.drawContours(image, [contour], -1, (0, 0, 255), 2)  # Vermelho para rachaduras
            cv2.putText(image, "Rachadura", tuple(contour[0][0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
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

    # Salvar a imagem processada automaticamente
    cv2.imwrite('resultado_processado.jpg', final_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
