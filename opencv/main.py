import cv2
import numpy as np

def detect_objects(frame):
    """
    Função para detectar objetos em um frame usando segmentação de cores e análise de contornos.
    """
   
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

   
    
    lower_bound = np.array([0, 50, 50]) 
    upper_bound = np.array([10, 255, 255])  

 
    mask = cv2.inRange(hsv, lower_bound, upper_bound)

   
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    
    for contour in contours:
        if cv2.contourArea(contour) > 50:  
          
            x, y, w, h = cv2.boundingRect(contour)

            
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "Objeto", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame, mask


def main():
    """
    Função principal para capturar vídeo e aplicar detecção de objetos.
    """
    
    cap = cv2.VideoCapture(1)  

    if not cap.isOpened():
        print("Erro ao acessar a câmera ou vídeo.")
        return

    while True:
       
        ret, frame = cap.read()
        if not ret:
            print("Não foi possível capturar o frame.")
            break

    
        processed_frame, mask = detect_objects(frame)

        
        cv2.imshow("Detecção de Objetos", processed_frame)
        cv2.imshow("Máscara", mask)

       
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
