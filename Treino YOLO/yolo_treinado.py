from ultralytics import YOLO
import cv2

image_path_to_test = "imagem_teste8.jpg" 
current_conf_to_test = 0.1   


model_path = "runs/detect/treino_rachaduras6/weights/best.pt"
try:
    model = YOLO(model_path)
except Exception as e:
    print(f"Erro ao carregar o modelo: {e}")
    print(f"Verifique se o caminho '{model_path}' está correto e o arquivo existe.")
    exit()

print(f"A testar com imagem: '{image_path_to_test}' e conf={current_conf_to_test}")

try:
    img_bgr = cv2.imread(image_path_to_test)
    if img_bgr is None:
        print(f"Erro: Não foi possível carregar a imagem em '{image_path_to_test}'.")
        print("Verifique se o arquivo existe e é uma imagem válida.")
        exit()
except Exception as e:
    print(f"Erro ao ler a imagem: {e}")
    exit()


training_imgsz = 640

try:
    results = model.predict(
        source=img_bgr,
        conf=current_conf_to_test, 
        imgsz=training_imgsz,
        augment=False
    )
except Exception as e:
    print(f"Erro durante a predição: {e}")
    exit()

if results and results[0].boxes is not None and len(results[0].boxes) > 0:
    print(f"Detectadas {len(results[0].boxes)} rachaduras.")
    results[0].show() 
    try:
        save_path = f"resultado_{image_path_to_test.split('.')[0]}_conf{current_conf_to_test}.jpg"
        results[0].save(save_path)
        print(f"Imagem com as detecções salva como '{save_path}'")
    except Exception as e:
        print(f"Erro ao salvar a imagem de resultado: {e}")
else:
    
    print(f"Nenhuma rachadura detectada com imgsz={training_imgsz} e conf={current_conf_to_test}.")
    print(f"Dimensões da imagem carregada: {img_bgr.shape}")

