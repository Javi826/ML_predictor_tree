import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Supongamos que tienes un DataFrame con tus datos
# X son las características y y es la variable objetivo que quieres predecir

# Dividir los datos en conjuntos de entrenamiento y prueba


# Construir el modelo de árbol de decisión
model = DecisionTreeClassifier()



# Entrenar el modelo
model.fit(X_train, y_train)

# Predecir en el conjunto de prueba
y_pred = model.predict(X_test)

# Calcular la precisión del modelo
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)