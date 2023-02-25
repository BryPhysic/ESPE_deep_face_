import time

import cv2

cap = cv2.VideoCapture(1)

leido, frame = cap.read()

if leido == True:
	cv2.imwrite(f"imgs_detect/foto{str(time.time())}.png", frame)
	print("Foto tomada correctamente")
else:
	print("Error al acceder a la cámara")

"""
	Finalmente liberamos o soltamos la cámara
"""
cap.release()