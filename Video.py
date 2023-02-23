import cv2
import time

cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Variable para controlar el tiempo de toma de fotos
start_time = time.time()

while True:
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:

        # Recortar la región del rostro dentro del rectángulo verde
        roi_color = frame[y:y+h, x:x+w]

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Tomar una foto de la región del rostro sin los bordes del rectángulo
        if time.time() - start_time >= 5:
            img_name = "foto_{}.png".format(str(time.time()).replace(".", ""))
            cv2.imwrite(img_name, roi_color)
            print("Foto guardada como: ", img_name)

            start_time = time.time()

    frame = cv2.flip(frame, 1)
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
