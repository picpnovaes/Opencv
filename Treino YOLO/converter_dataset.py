import os
import cv2
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import shutil

# ========= CONFIGURAR AQUI =========
ORIGEM = Path(r"C:\Users\Pedro\OneDrive\Área de Trabalho\Projeto 3- YOLO\datasets\SDNET2018\DATA_Maguire_20180517_ALL\W\CW")
DESTINO_BASE = Path(r"C:\Users\Pedro\OneDrive\Área de Trabalho\Projeto 3- YOLO\datasets\SDNET2018_YOLO_FINAL")
CLASS_ID = 0  
# ===================================

def criar_estrutura():
    """Cria a estrutura de pastas YOLO"""
    for subset in ['train', 'val']:
        for folder in ['images', 'labels']:
            (DESTINO_BASE / subset / folder).mkdir(parents=True, exist_ok=True)

def gerar_anotacao_yolo(img_path, txt_path):
    """Gera anotação YOLO simulada (substitua pela sua lógica real)"""
    altura, largura = 640, 640  
    x_center, y_center = 0.5, 0.5  
    w, h = 0.3, 0.3  
    
    with open(txt_path, 'w') as f:
        f.write(f"{CLASS_ID} {x_center} {y_center} {w} {h}\n")

def processar_dataset():
    imagens = list(ORIGEM.glob('*.jpg'))
    if not imagens:
        print(f"❌ Nenhuma imagem encontrada em {ORIGEM}")
        return

    train, val = train_test_split(imagens, test_size=0.2, random_state=42)

    print("🔄 Processando TRAIN...")
    for img_path in train:

        dest_img = DESTINO_BASE / 'train' / 'images' / img_path.name
        shutil.copy2(img_path, dest_img)
        
        dest_txt = DESTINO_BASE / 'train' / 'labels' / f"{img_path.stem}.txt"
        gerar_anotacao_yolo(img_path, dest_txt)

    print("🔄 Processando VAL...")
    for img_path in val:
        dest_img = DESTINO_BASE / 'val' / 'images' / img_path.name
        shutil.copy2(img_path, dest_img)
        
        dest_txt = DESTINO_BASE / 'val' / 'labels' / f"{img_path.stem}.txt"
        gerar_anotacao_yolo(img_path, dest_txt)

    print(f"✅ Concluído! Total de imagens processadas: {len(imagens)}")
    print(f"📂 Estrutura criada em: {DESTINO_BASE}")

if __name__ == "__main__":
    criar_estrutura()
    processar_dataset()