# librerias 
import pandas as pd

from sklearn.preprocessing import LabelEncoder

# Leer el archivo de datos
df = pd.read_csv('Datos\Datos_cosumo.csv')

# Eliminar duplicados
df = df.drop_duplicates()

# Tratar valores faltantes
df = df.dropna()  # Eliminar filas con valores faltantes

# Eliminar valores atípicos
df = df[(df['Edad'] > 0) & (df['Edad'] < 120)]  # Eliminar valores de edad imposibles

# Corregir errores tipográficos o de formato
df['Genero'] = df['Genero'].str.lower()  # Convertir a minúsculas

#ver las categorias de las variables
# Leer el archivo de datos
#df = pd.read_csv('datos.csv')


# Crear un objeto LabelEncoder
le = LabelEncoder()

# Crear diccionario con valores originales
cat_cols = ['Genero', 'Rasgos', 'Mes', 'Ciudad', 'Estacion', 'Dia Festivo', 'Tipo de Refresco', 'Region de la Ciudad']
cat_dict = {}
for col in cat_cols:
    cat_dict[col] = list(df[col].unique())

# Aplicar LabelEncoder a cada variable categórica
for col in cat_cols:
    df[col] = le.fit_transform(df[col])
    cat_dict[col] = {k: v for k, v in sorted({le.transform([val])[0]: val for val in cat_dict[col]}.items())}

# Imprimir el diccionario con valores codificados
#
# print(cat_dict)

# Codificar las variables categóricas
df['Genero'] = le.fit_transform(df['Genero'])
df['Rasgos'] = le.fit_transform(df['Rasgos'])
df['Mes'] = le.fit_transform(df['Mes'])
df['Ciudad'] = le.fit_transform(df['Ciudad'])
df['Estacion'] = le.fit_transform(df['Estacion'])
df['Dia Festivo'] = le.fit_transform(df['Dia Festivo'])
df['Tipo de Refresco'] = le.fit_transform(df['Tipo de Refresco'])
df['Region de la Ciudad'] = le.fit_transform(df['Region de la Ciudad'])

# Guardar el archivo limpio y codificado
df.to_csv('datos_limpios.csv', index=False)

##modelos de  regrrecion logistica
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

# Seleccionar las características y la variable objetivo
X = df.drop('Tipo de Refresco', axis=1)
y = df['Tipo de Refresco']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear un objeto LogisticRegression
# Estándarizar los datos de entrada
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Entrenar el modelo con los datos de entrenamiento estandarizados
clf = LogisticRegression()
clf.fit(X_train, y_train)

# Predecir los valores del conjunto de prueba
y_pred = clf.predict(X_test)

# Evaluar la precisión del modelo
accuracy = clf.score(X_test, y_test)
print('Precisión del modelo_stadarizado: {:.2f}'.format(accuracy))


# Normalizar los datos
scaler = MinMaxScaler()
X_train_norm = scaler.fit_transform(X_train)
X_test_norm = scaler.transform(X_test)
# Entrenar un modelo de regresión logística
clf = LogisticRegression(random_state=42)
clf.fit(X_train_norm, y_train)

# Predecir el tipo de refresco en el conjunto de prueba
y_pred = clf.predict(X_test_norm)

# Evaluar la precisión del modelo
accuracy = accuracy_score(y_test, y_pred)
print("Precisión del modelo normalizado : {:.2f}".format(accuracy))

# no son valores muy buenos 
import joblib
joblib.dump(clf, 'models/modelo_entrenado.joblib')

clf_cargado = joblib.load('models/modelo_entrenado.joblib')


