import cv2
import numpy as np

def detect_cracks(frame):
    """Detect cracks in the given frame."""
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

   
    edges = cv2.Canny(blurred, 50, 150)

   
    kernel = np.ones((3, 3), np.uint8)
    edges_dilated = cv2.dilate(edges, kernel, iterations=1)

   
    contours, _ = cv2.findContours(edges_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) > 50:  
            cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)  
    return frame, edges

def detect_objects_and_cracks(frame):
    """Detect objects and cracks in the frame."""
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_bound = np.array([0, 50, 50])
    upper_bound = np.array([10, 255, 255])
    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) > 500:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "Objeto", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

   
    frame_with_cracks, edges = detect_cracks(frame)

    return frame_with_cracks, mask, edges


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

  
    processed_frame, mask, edges = detect_objects_and_cracks(frame)

  
    cv2.imshow("Frame - Objetos e Rachaduras", processed_frame)
    cv2.imshow("Mask - Objetos", mask)
    cv2.imshow("Edges - Rachaduras", edges)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
