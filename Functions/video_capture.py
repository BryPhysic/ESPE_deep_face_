import cv2
import time

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')

# Variable para controlar la toma de fotos
take_photo = False
take_photo_f = False
while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5) 
    # Inicializar roi_color fuera del bucle for
    roi_color = None

    for (x, y, w, h) in faces:
        # Recortar la región del rostro dentro del rectángulo verde
        roi_color = frame[y:y+h, x:x+w]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Tomar una foto de toda la imagen con los bordes del rectángulo
    if cv2.waitKey(1) == ord('t'):
        roi_color = frame
        take_photo_f = True
        if take_photo_f and roi_color is not None: # Verificar si se ha detectado algún rostro
            img_name = "imgs_detect/foto_{}_full.png".format(str(time.time()).replace(".", ""))
            cv2.imwrite(img_name, roi_color)
            print("Foto guardada como: ", img_name)
            take_photo = False


    # Tomar una foto de la región del rostro sin los bordes del rectángulo
    if cv2.waitKey(1) == ord('c'):
        take_photo = True

    if take_photo and roi_color is not None: # Verificar si se ha detectado algún rostro
        img_name = "imgs_detect/foto_{}.png".format(str(time.time()).replace(".", ""))
        cv2.imwrite(img_name, roi_color)
        print("Foto guardada como: ", img_name)
        take_photo = False

    frame = cv2.flip(frame, 1)
    cv2.imshow('frame', frame)

    # Presionar 'q' para salir del bucle
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()

