# creo un data set  fictisio para  poder  hacer  pruebas
#%%
import pandas as pd
import random
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
    'Montecristi': 'Costa',
    'Pedernales': 'Costa',
    'Cotacachi': 'Sierra',
    'Puerto López': 'Costa',
    'San Lorenzo': 'Costa',
    'Calceta': 'Costa',
    'Rocafuerte': 'Costa',
    'Atuntaqui': 'Sierra',
    'Yaguachi': 'Costa',
    'Banos': 'Costa',
    'Balzar': 'Costa',
    'Daule': 'Costa',
    'Cayambe': 'Sierra',
    'Gualaceo': 'Sierra',
    'Pelileo': 'Sierra',
    'Pinas': 'Costa',
    'Catamayo': 'Sierra',
    'Rosa Zarate': 'Costa',
    'Santa Ana': 'Costa',
    'Sucre': 'Costa',
    'Bolivar': 'Sierra',
    'Cariamanga': 'Sierra',
    'El Carmen': 'Costa',
    'Gonzanama': 'Sierra',
    'La Mana': 'Sierra',
    'La Troncal': 'Costa',
    'Muisne': 'Costa',
    'Naranjal': 'Costa',
    'San Vicente': 'Costa',
    'Yantzaza': 'Oriente'
}
#%%
def generar_datos_aleatorios(num_filas):
    edades = [random.randint(10, 70) for _ in range(num_filas)]
    generos = [random.choice(['Man', 'Woman']) for _ in range(num_filas)]
    razas = [random.choice(['asian', 'indian', 'black', 'white', 'middle eastern', 'latino hispanic']) for _ in range(num_filas)]
    horas = [random.randint(8, 22) for _ in range(num_filas)]
    ciudades = [
                                                    'Quito',
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
                                                    'Baños de Agua Santa',
                                                    'Montecristi',
                                                    'Pedernales',
                                                    'Cotacachi',
                                                    'Puerto López',
                                                    'San Lorenzo',
                                                    'Calceta',
                                                    'Rocafuerte',
                                                    'Atuntaqui',
                                                    'Yaguachi',
                                                    'Banos',
                                                    'Balzar',
                                                    'Daule',
                                                    'Cayambe',
                                                    'Gualaceo',
                                                    'Pelileo',
                                                    'Pinas',
                                                    'Catamayo',
                                                    'Rosa Zarate',
                                                    'Santa Ana',
                                                    'Sucre',
                                                    'Bolivar',
                                                    'Cariamanga',
                                                    'El Carmen',
                                                    'Gonzanama',
                                                    'La Mana',
                                                    'La Troncal',
                                                    'Muisne',
                                                    'Naranjal',
                                                    'San Vicente',
                                                    'Yantzaza'
                                                ]
    estaciones = [random.choice(['Invierno', 'Verano']) for _ in range(num_filas)]
    dias_festivos = [random.choice([True, False]) for _ in range(num_filas)]
    tipos_refresco = ['Coca-Cola', 'Pepsi', 'Fanta', 'Sprite', '7-Up', 'Guarana', 'Tonica', 'Ginger Ale', 'Jamaica', 'Agua mineral']
    datos = {'Edad': edades, 'Genero': generos, 'Raza': razas, 'Hora': horas, 'Ciudad': [random.choice(ciudades) for _ in range(num_filas)],
              'Estacion': estaciones, 'Dia Festivo': dias_festivos,
             'Tipo de Refresco': [random.choice(tipos_refresco) for _ in range(num_filas)]}
    datos['Region de la Ciudad'] = [ciudades_ecuador[ciudad] for ciudad in datos['Ciudad']]
    return pd.DataFrame(datos)

#%%

datos_aleatorios = generar_datos_aleatorios(17000000)
# %%
datos_aleatorios.to_csv('datos_aleatorios.csv', index=False)
# %%
datos_aleatorioss = generar_datos_aleatorios(1000)
datos_aleatorioss.to_csv('datos_aleatorioss.csv', index=False)
# %%
