import cv2
import streamlit as st
import time
import pandas as pd
from deepface import DeepFace
import joblib
import pickle
from sklearn.preprocessing import MinMaxScaler
import numpy as np
global frame, roi_color, data, data_portada
def main():
    global frame, roi_color, data, data_portada
    Menu = st.sidebar.selectbox("Filtro", ["Portada", "Take a Picture", "Clasificate"])
    data = {}
    if Menu == "Portada":
        data = pd.read_csv('Datos\Datos_cosumo.csv')

        # Mostrar las columnas y sus posibles valores
        for col in data.columns:
            st.write(f"**{col}**")
            st.write(data[col].unique())
        
    elif Menu == "Take a Picture":
        st.title("Reconocer rostro")
        cap = cv2.VideoCapture(0)
        face_cascade = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')

        if not cap.isOpened():
            st.error("No se pudo abrir la cámara")
            return
        img = None
        stframe = st.empty()
        col1, col2 = st.columns(2)

        with col1:
            capture_button = st.button("Tomar captura")
        with col2:
            capture_button_f = st.button("Tomar captura solo el rostro")


        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("Error al recibir imagen de la cámara")
                break

            # Convierte el frame de BGR a RGB para mostrarlo en Streamlit
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = face_cascade.detectMultiScale(frame, 1.3, 5)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                if capture_button_f: # quiero que solo capture los pixeles que estan dentro del cuadro verde
                    roi_color = frame[y:y+h, x:x+w]
                    #st.image(roi_color, caption="Imagen capturada")
                    roi_color = cv2.cvtColor(roi_color, cv2.COLOR_BGR2RGB)
                    img_name = "imgs_detect/Photo_face.jpg"
                    img=cv2.imwrite(img_name, roi_color)
                    st.success("¡Rostro encontrado!")
                    output =  DeepFace.analyze(img_path = 'imgs_detect/Photo_face.jpg',actions = ['age', 'gender', 'race', 'emotion'])
                    data={'Edad':output[0]['age'],
                      'Genero':output[0]['dominant_gender'],
                      'Rasgos':output[0]['dominant_race'],
                      'Emocion':output[0]['dominant_emotion']
                        }
                    data_portada = data
                    st.info('Según el análisis de la imagen:\n')
                    col1, col2 = st.columns(2)
    
                    with col1:
                        roi_color = cv2.cvtColor(roi_color, cv2.COLOR_BGR2RGB)
                        st.image(roi_color, caption="Imagen capturada",use_column_width=True)                
                    
                    with col2:
                        
                        st.table(data)
                    capture_button_f = False
                    break

            frame = cv2.flip(frame, 1)
            # Mostrar el frame en Streamlit
            
            if capture_button:
                # Guardar la imagen capturada
                frame = frame
                st.image(frame, caption="Imagen capturada")
                img_name = "imgs_detect/Photo.jpg"
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                cv2.imwrite(img_name, frame)
                st.success("¡Captura completa guardada con éxito!")
                output =  DeepFace.analyze(img_path = 'imgs_detect/Photo.jpg',actions = ['age', 'gender', 'race', 'emotion'])
                data={'Edad':output[0]['age'],
                      'Genero':output[0]['dominant_gender'],
                      'Rasgos':output[0]['dominant_race'],
                      'Emocion':output[0]['dominant_emotion']
                        }
                        
                data_portada = data
                st.info('Según el análisis de la imagen:\n')
                st.table(data)
                


                capture_button = False
            #st.image(img, caption="Imagen capturada")
            stframe.image(frame)
        cap.release()

    elif Menu == "Clasificate":

        
       
        ciudades_ecuador = {
                                'Quito': 'Sierra',
                                'Guayaquil': 'Costa',
                                'Cuenca': 'Sierra',
                                'Ambato': 'Sierra',
                                'Manta': 'Costa',
                                'Esmeraldas': 'Costa',
                                'Ibarra': 'Sierra',
                                'Loja': 'Sierra',
                                'Portoviejo': 'Costa',
                                'Riobamba': 'Sierra',
                                'Salinas': 'Costa',
                                'Santa Elena': 'Costa',
                                'Tena': 'Oriente',
                                'Machala': 'Costa',
                                'Playas': 'Costa',
                                'Babahoyo': 'Costa',
                                'Machachi': 'Sierra',
                                'Latacunga': 'Sierra',
                                'Puyo': 'Oriente',
                                'Zamora': 'Oriente',
                                'Coca': 'Oriente',
                                'Puerto Francisco de Orellana': 'Oriente',
                                'Macas': 'Oriente',
                                'Tulcan': 'Sierra',
                                'Santo Domingo de los Tsachilas': 'Costa',
                                'Guaranda': 'Sierra',
                                'Azogues': 'Sierra',
                                'Chone': 'Costa',
                                'La Libertad': 'Costa',
                                'Pasaje': 'Costa',
                                'Samborondon': 'Costa',
                                'Milagro': 'Costa',
                                'Vinces': 'Costa',
                                'Jipijapa': 'Costa',
                                'Santa Rosa': 'Costa',
                                'San Gabriel': 'Sierra',
                                'Nueva Loja': 'Oriente',
                                'Puerto Ayora': 'Galapagos',
                                'Puerto Villamil': 'Galapagos',
                                'Baños de Agua Santa': 'Sierra',
                                
                            }
        
        Edad = st.slider('Seleccione una edad', 18, 70, 18)
        Genero = st.selectbox('Seleccione un género', ['Man', 'Woman'])
        Rasgos = st.selectbox('Seleccione un rasgo', ['asian', 'indian', 'black', 'white', 'middle eastern', 'latino hispanic'])
        emocion = st.selectbox('Seleccione una emoción', ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'])
        meses = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio', 'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre']

        Mes = st.selectbox('Mes de compra', meses)
        Hora = st.slider('hora', 8, 22, 13)
        ciudades = [    'Quito',
                        'Guayaquil',
                        'Cuenca',
                        'Ambato',
                        'Manta',
                        'Esmeraldas',
                        'Ibarra',
                        'Loja',
                        'Portoviejo',
                        'Riobamba',
                        'Salinas',
                        'Santa Elena',
                        'Tena',
                        'Machala',
                        'Playas',
                        'Babahoyo',
                        'Machachi',
                        'Latacunga',
                        'Puyo',
                        'Zamora',
                        'Coca',
                        'Puerto Francisco de Orellana',
                        'Macas',
                        'Tulcan',
                        'Santo Domingo de los Tsachilas',
                        'Guaranda',
                        'Azogues',
                        'Chone',
                        'La Libertad',
                        'Pasaje',
                        'Samborondon',
                        'Milagro',
                        'Vinces',
                        'Jipijapa',
                        'Santa Rosa',
                        'San Gabriel',
                        'Nueva Loja',
                        'Puerto Ayora',
                        'Puerto Villamil',
                        'Baños de Agua Santa']

        Ciudad = st.selectbox('Seleccione una ciudad', ciudades)
        
        Region = st.write(f'Region : {ciudades_ecuador[Ciudad]}')
        Estacion = st.selectbox('Seleccione una estación', ['Invierno', 'Verano'])
        Dia_Festivo= st.checkbox('¿Es día festivo?')
        
        if st.button('Predecir'):

            with open("models\label_encoders.pkl", "rb") as f:
                le_list = pickle.load(f)
            with open('models\modelo.pkl', 'rb') as archivo:
                model = pickle.load(archivo)

            data = {
                'Edad': Edad,
                'Genero': Genero,
                'Rasgos': Rasgos,
                'Emocion': emocion,
                'Mes': Mes,
                'Hora': Hora,
                'Ciudad': Ciudad,
                'Region': ciudades_ecuador[Ciudad],
                'Estacion': Estacion,
                'Dia_Festivo': Dia_Festivo
            }

            data = pd.DataFrame(data, index=[0])
            st.write(data)

            data2 = {
                'Edad': Edad,
                'Genero': Genero,
                'Rasgos': Rasgos,
                
                'Mes': Mes,
                'Hora': Hora,
                'Ciudad': Ciudad,
                'Estacion': Estacion,
                'Dia_Festivo':  True if Dia_Festivo else False,
                'Region': ciudades_ecuador[Ciudad]
                
                
            }

            dfn = pd.DataFrame(data2, index=[0])
            dfn['Edad'] = dfn['Edad'].astype(int)
            dfn['Hora'] = dfn['Hora'].astype(int)
            dfn["Genero"] = le_list[0].transform([dfn["Genero"]])[0]
            dfn["Rasgos"] = le_list[1].transform([dfn["Rasgos"]])[0]
            dfn["Mes"] = le_list[2].transform([dfn["Mes"]])[0]
            dfn["Ciudad"] = le_list[3].transform([dfn["Ciudad"]])[0]
            dfn["Estacion"] = le_list[4].transform([dfn["Estacion"]])[0]
            dfn["Dia_Festivo"] = le_list[5].transform([dfn["Dia_Festivo"]])[0]
            #dfn["Dia_Festivo"] = le_list[5].transform([eval(dfn["Dia_Festivo"][0])])[0]
            dfn["Region"] = le_list[6].transform([dfn["Region"]])[0]

            st.write(dfn)
            prediccion = model.predict(dfn)
            val = le_list[7].inverse_transform([prediccion[0]])
            st.success(f'El refresco que le conviene es: {val}')


            print(data)
            
            

            #data = scaler.fit_transform(data)
            #prediction = model.predict(data)






if __name__ == "__main__":
    main()
