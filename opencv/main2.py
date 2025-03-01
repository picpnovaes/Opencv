import cv2
import numpy as np


def detect_cracks(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    

    edges = cv2.Canny(blurred, 50, 150)
    

    kernel = np.ones((3, 3), np.uint8)
    edges_dilated = cv2.dilate(edges, kernel, iterations=1)
    

    contours, _ = cv2.findContours(edges_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    

    for contour in contours:
        if cv2.contourArea(contour) > 50:  
            cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)
    
    return image, edges


image_path = "superficie.jpg"  
image = cv2.imread(image_path)


result_image, edges = detect_cracks(image)

cv2.imshow("Rachaduras Detectadas", result_image)
cv2.imshow("Bordas", edges)

cv2.waitKey(0)
cv2.destroyAllWindows()
