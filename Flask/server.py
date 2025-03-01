from flask import Flask, request, send_file
import cv2
import numpy as np
import os
from werkzeug.utils import secure_filename
from time import time

app = Flask(__name__)

# Configurações do diretório de uploads e limite de tamanho de arquivo
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # Limite de 16 MB para uploads

# Defina as extensões de arquivo permitidas
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    """Verifica se o arquivo tem uma extensão permitida."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def detect_cracks(image):
    """Processa a imagem para detectar rachaduras."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        if cv2.contourArea(contour) > 50:
            cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)

    return image

@app.route("/upload", methods=["POST"])
def upload_file():
    """Recebe uma imagem, processa e retorna a imagem com as rachaduras detectadas."""
    if "file" not in request.files:
        return "Nenhum arquivo enviado", 400
    
    file = request.files["file"]
    if file.filename == "":
        return "Nenhum arquivo selecionado", 400
    
    # Verifica se o tipo de arquivo é permitido
    if not allowed_file(file.filename):
        return "Tipo de arquivo não permitido", 400
    
    # Gera um nome único para o arquivo usando timestamp
    filename = f"{int(time())}_{secure_filename(file.filename)}"
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)
    
    image = cv2.imread(filepath)
    if image is None:
        return "Erro ao carregar a imagem", 400
    
    processed_image = detect_cracks(image)
    
    # Gera o nome do arquivo processado com um prefixo
    processed_filename = f"processed_{filename}"
    processed_filepath = os.path.join(app.config["UPLOAD_FOLDER"], processed_filename)
    cv2.imwrite(processed_filepath, processed_image)
    
    return send_file(processed_filepath, mimetype="image/jpeg")

if __name__ == "__main__":
    app.run(debug=True)
