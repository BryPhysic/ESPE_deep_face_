
#aplicamos  grid search para encontrar los mejores parametros
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# Cargar los datos
datos = pd.read_csv("Datos\Datos_cosumo.csv")

# Codificar las variables categóricas
le = LabelEncoder()
datos['Genero'] = le.fit_transform(datos['Genero'])
datos['Rasgos'] = le.fit_transform(datos['Rasgos'])
datos['Mes'] = le.fit_transform(datos['Mes'])
datos['Hora'] = le.fit_transform(datos['Hora'])
datos['Ciudad'] = le.fit_transform(datos['Ciudad'])
datos['Estacion'] = le.fit_transform(datos['Estacion'])
datos['Dia Festivo'] = le.fit_transform(datos['Dia Festivo'])
datos['Region de la Ciudad'] = le.fit_transform(datos['Region de la Ciudad'])

# Dividir los datos en conjuntos de entrenamiento y prueba
X = datos.drop(['Tipo de Refresco'], axis=1)
y = datos['Tipo de Refresco']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizar los datos
scaler = StandardScaler()
X_train_norm = scaler.fit_transform(X_train)
X_test_norm = scaler.transform(X_test)

models = [
    {
        'name': 'Logistic Regression',
        'estimator': LogisticRegression(),
        'hyperparameters': {
            'solver': ['newton-cg', 'lbfgs', 'liblinear'],
            'penalty': ['l2'],
            'C': [0.1, 0.5, 0.9, 1, 10, 150, 500],
            'max_iter': [100, 500, 1000]
        }
    },
    {
        'name': 'Decision Tree',
        'estimator': DecisionTreeClassifier(),
        'hyperparameters': {
            'criterion': ['entropy', 'gini'],
            'max_depth': [None, 2, 5, 10],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 5]
        }
    },
    {
        'name': 'Random Forest',
        'estimator': RandomForestClassifier(),
        'hyperparameters': {
            'n_estimators': [10, 50, 100],
            'criterion': ['entropy', 'gini'],
            'max_depth': [None, 2, 5, 10],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 5]
        }
    },
    {
        'name': 'Support Vector Machine',
        'estimator': SVC(),
        'hyperparameters': {
            'kernel': ['linear', 'rbf'],
            'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
            'gamma': ['scale', 'auto']
        }
    }
]

# Buscar los mejores hiperparámetros para cada modelo usando GridSearchCV
for model in models:
    print(f"Model: {model['name']}")
    print(f"Tuning hyperparameters...")
    grid = GridSearchCV(model['estimator'], model['hyperparameters'], cv=5)
    grid.fit(X_train_norm, y_train)
    print(f"Best parameters: {grid.best_params_}")
    print(f"Training accuracy: {grid.best_score_:.2f}")
    y_pred = grid.predict(X_test_norm)
    test_accuracy = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {test_accuracy:.2f}")
    print("-" * 30)


