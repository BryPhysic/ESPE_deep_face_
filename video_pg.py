import streamlit as st
import cv2
import time

# Configuración de la página de Streamlit
st.set_page_config(
    page_title="Streamlit OpenCV Example",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Crear una sección de la barra lateral para mostrar las fotos guardadas
st.sidebar.title("Fotos guardadas")
uploaded_files = st.sidebar.empty()

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Crear un espacio para mostrar la imagen de la cámara
image_space = st.empty()

# Agregar un botón para tomar una foto
if st.button('Tomar foto'):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        roi_color = frame[y:y+h, x:x+w]
        img_name = "foto_{}.png".format(str(time.time()).replace(".", ""))
        cv2.imwrite(img_name, roi_color)
        print("Foto guardada como: ", img_name)
        uploaded_files.image(img_name, width=100)
    image_space.image(frame, channels="BGR")

cap.release()
cv2.destroyAllWindows()
